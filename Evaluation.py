"""Quantitative closed-loop evaluation for autopilot models.

Offline validation loss has repeatedly disagreed with how a model actually drives --
three separate metrics (MAE, mixed-camera val_steer, center-only val_steer) each ranked
models in the wrong order. The root cause is that labels come from a keyboard, so a
model that steers smoothly scores as "wrong" against bang-bang demonstrations.

This module measures the thing we actually care about: put the model in the driving
seat, let it drive, and record what happens. Every run uses the same world seed, so
two models face an identical track, identical obstacle placement and identical signs.

Usage: menu -> "Evaluate Model", or Evaluation.startEvaluationFlow(worldGen).
"""
#note: why tf was I planning on doing this manually, this would've taken so much longer

import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from ursina import *

import Autopilot_sim


# The driving lane's centre sits at roadCentreline + trueRight * (roadWidth * 0.25).
# Derived from StopSign.groundStrip: the sign is placed at centre + right*(w/2 + 1.5)
# and the strip hangs at local x -(w/4 + 1.5) from it, landing at centre + right*(w/4).
LANE_CENTRE_FRACTION = 0.25

# NOTE: generateRoad()'s local variable named `right` is cross(forward, UP), which for
# forward=+Z gives -X -- i.e. it actually points LEFT. That is why object placement
# throughout Autopilot_sim negates its x component. We compute the true right vector
# here rather than inherit that quirk.
GLOBAL_UP = np.array([0.0, 1.0, 0.0])

EVAL_WORLD_SEED = 20260808   # fixed so every model drives the same course

# Speed is sampled as the car crosses each of these distances from a stop sign. This
# replaces "distance at which the brake was first applied", which turned out unusable:
# a model that brakes often is already braking when a sign comes into range, so nearly
# every sign got excluded. The profile needs no event detection -- a curve that stays
# flat until 20 units means late detection, one that declines from 100 means it saw it.
APPROACH_BUCKETS = (120, 100, 80, 60, 40, 30, 20, 10)

# Where the car must come to rest for a stop to register at all.
# StopSign.trigger is scale (roadWidth, 10, 14) at local (-(w/4+1.5), 2, -8), so it
# spans local z -15..-1. The sign entity sits at roadCentre + right*(w/2 + 1.5) while
# the driving lane is at roadCentre + right*(w/4), leaving 4.0 units of lateral
# separation. Planar distance from a car in-lane to the sign is therefore
# sqrt(4^2 + z^2): 4.12 units at the near edge, 15.52 at the far edge.
# Stopping closer than 4.12 means it rolled PAST the zone -- the sign never turns
# green, and the car sits there. Observed in testing.
STOP_ZONE_NEAR = 4.12
STOP_ZONE_FAR = 15.52


def _trueRight(forward):
    r = np.cross(GLOBAL_UP, forward)
    n = math.sqrt(float(r @ r))
    return r / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


class LaneGeometry:
    """Nearest-point queries against the road spline.

    trackPoints mutates as the world scrolls (updateTrack pops the front and appends a
    new milestone), so the sampled centreline is rebuilt whenever that happens.
    """

    def __init__(self, worldGen, samplesPerSegment=24):
        self.worldGen = worldGen
        self.samplesPerSegment = samplesPerSegment
        self.roadWidth = getattr(worldGen, "globalRoadWidth", 10)
        self._signature = None
        self.centres = np.zeros((0, 3))
        self.rights = np.zeros((0, 3))
        self.forwards = np.zeros((0, 3))
        self.rebuildIfStale()

    def _currentSignature(self):
        tp = self.worldGen.trackPoints
        return (len(tp), tuple(tp[0]), tuple(tp[-1]))

    def rebuildIfStale(self):
        sig = self._currentSignature()
        if sig == self._signature:
            return
        self._signature = sig

        tp = self.worldGen.trackPoints
        centres, rights, forwards = [], [], []
        for i in range(1, len(tp) - 2):
            p0, p1, p2, p3 = tp[i - 1], tp[i], tp[i + 1], tp[i + 2]
            for j in range(self.samplesPerSegment + 1):
                t = j / self.samplesPerSegment
                c = Autopilot_sim.spline(p0, p1, p2, p3, t)
                f = Autopilot_sim.splineTan(p0, p1, p2, p3, t)
                centres.append(np.asarray(c, dtype=float))
                forwards.append(np.asarray(f, dtype=float))
                rights.append(_trueRight(np.asarray(f, dtype=float)))
        self.centres = np.array(centres)
        self.rights = np.array(rights)
        self.forwards = np.array(forwards)

    def laneCentreOffsetVector(self, position):
        """Vector to add to `position` to place it on the driving-lane centre.

        The car spawns at Vec3(0, 10, 0), which is the road CENTRELINE -- half a lane
        out of position. Left uncorrected, every model starts straddling the yellow
        line and logs a departure in its first few units.
        """
        self.rebuildIfStale()
        if len(self.centres) == 0:
            return Vec3(0, 0, 0)
        p = np.array([position.x, position.y, position.z])
        planar = (self.centres[:, 0] - p[0]) ** 2 + (self.centres[:, 2] - p[2]) ** 2
        k = int(np.argmin(planar))
        lateral = float((p - self.centres[k]) @ self.rights[k])
        shift = (self.roadWidth * LANE_CENTRE_FRACTION) - lateral
        r = self.rights[k]
        return Vec3(float(r[0] * shift), 0.0, float(r[2] * shift))

    def curvatureAt(self, position):
        """Heading change in degrees over ~50 units of road near `position`.

        Tests the straightaway hypothesis directly: a sign on a curve drifts sideways in
        the frame during the approach and can leave the crop entirely.
        """
        self.rebuildIfStale()
        if len(self.forwards) < 6:
            return 0.0
        p = np.array([position.x, position.y, position.z])
        planar = (self.centres[:, 0] - p[0]) ** 2 + (self.centres[:, 2] - p[2]) ** 2
        k = int(np.argmin(planar))
        # +-4 samples ~= 67 units. Narrower windows read zero at an S-bend inflection,
        # where the road is momentarily straight between two opposite curves.
        a = self.forwards[max(0, k - 4)].copy()
        b = self.forwards[min(len(self.forwards) - 1, k + 4)].copy()
        a[1] = b[1] = 0.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        dot = max(-1.0, min(1.0, float((a / na) @ (b / nb))))
        return abs(math.degrees(math.acos(dot)))

    def query(self, position, carYawDegrees):
        """Return (lateralOffset, headingErrorDeg) or None.

        lateralOffset is signed distance right of the road centreline, so the driving
        lane spans 0..roadWidth/2 and its centre is roadWidth*0.25.
        """
        self.rebuildIfStale()
        if len(self.centres) == 0:
            return None

        p = np.array([position.x, position.y, position.z])
        planar = (self.centres[:, 0] - p[0]) ** 2 + (self.centres[:, 2] - p[2]) ** 2
        k = int(np.argmin(planar))

        delta = p - self.centres[k]
        lateral = float(delta @ self.rights[k])

        yaw = math.radians(carYawDegrees)
        carForward = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        roadForward = self.forwards[k].copy()
        roadForward[1] = 0.0
        n = math.sqrt(float(roadForward @ roadForward))
        if n < 1e-9:
            return lateral, 0.0
        roadForward /= n

        dot = max(-1.0, min(1.0, float(carForward @ roadForward)))
        cross = carForward[2] * roadForward[0] - carForward[0] * roadForward[2]
        headingErr = math.degrees(math.acos(dot)) * (1.0 if cross >= 0 else -1.0)
        return lateral, headingErr


class EvaluationRun(Entity):
    """Drives the model for a fixed wall-clock duration and records behaviour."""

    def __init__(self, worldGen, duration=90.0, sampleRate=20.0, onFinish=None, **kwargs):
        super().__init__(**kwargs)
        self.worldGen = worldGen
        self.duration = duration
        self.samplePeriod = 1.0 / sampleRate
        self.onFinish = onFinish

        self.geometry = LaneGeometry(worldGen)
        self.laneWidth = self.geometry.roadWidth / 2.0
        self.laneCentre = self.geometry.roadWidth * LANE_CENTRE_FRACTION

        self.elapsed = 0.0
        self.sampleTimer = 0.0
        self.finished = False

        self.laneDeviations = []      # |offset from driving-lane centre|
        self.headingErrors = []
        self.speeds = []
        self.speedLimits = []
        self.wheelAngles = []

        self.distance = 0.0
        self.distanceBeforeFirstDeparture = None
        self.lastPosition = Vec3(worldGen.car.position)

        self.offRoadSamples = 0
        self.wrongLaneSamples = 0
        self.totalSamples = 0
        self.departures = 0
        self._wasOnRoad = True

        self.pedestrianStrikes = 0
        self._pedestrianCooldown = 0.0

        # Manual rescues invalidate part of a run -- record how much of it was yours.
        self.interventionSamples = 0

        # Recovery: leaving the lane once and never getting back reads the same as one
        # brief wobble in the departure count, but they are completely different
        # failures. Timed from leaving the road to regaining it, and rescues are tracked
        # separately so a recovery you performed is never credited to the model.
        self.recoveryTimes = []
        self.recoveriesHelped = 0
        self.neverRecovered = 0
        self._departureStart = None
        self._departureHelped = False

        self.stopSigns = {}           # id(sign) -> record

    # ------------------------------------------------------------------ sampling

    def update(self):
        if self.finished or not self.enabled:
            return

        car = self.worldGen.car
        moved = Vec3(car.position) - self.lastPosition
        self.distance += math.sqrt(moved.x ** 2 + moved.z ** 2)
        self.lastPosition = Vec3(car.position)

        self.elapsed += time.dt
        self._pedestrianCooldown = max(0.0, self._pedestrianCooldown - time.dt)

        self.sampleTimer += time.dt
        if self.sampleTimer >= self.samplePeriod:
            self.sampleTimer = 0.0
            self._takeSample()

        if self.elapsed >= self.duration:
            self.finish()

    def _takeSample(self):
        car = self.worldGen.car
        q = self.geometry.query(car.position, car.rotation_y)
        if q is None:
            return
        lateral, headingErr = q

        self.totalSamples += 1
        if any(held_keys[k] for k in ("w", "a", "s", "d")):
            self.interventionSamples += 1
        self.laneDeviations.append(abs(lateral - self.laneCentre))
        self.headingErrors.append(abs(headingErr))
        self.speeds.append(abs(car.returnVelocity()))
        self.speedLimits.append(self.worldGen.returnSpeedLimit())
        self.wheelAngles.append(car.wheelAngle)

        onRoad = -self.laneWidth <= lateral <= self.laneWidth
        inCorrectLane = 0.0 <= lateral <= self.laneWidth
        if not onRoad:
            self.offRoadSamples += 1
        elif not inCorrectLane:
            self.wrongLaneSamples += 1

        humanNow = any(held_keys[k] for k in ("w", "a", "s", "d"))

        if self._wasOnRoad and not onRoad:
            self.departures += 1
            self._departureStart = self.elapsed
            self._departureHelped = humanNow
            if self.distanceBeforeFirstDeparture is None:
                self.distanceBeforeFirstDeparture = self.distance
        elif not self._wasOnRoad:
            if humanNow:
                self._departureHelped = True
            if onRoad and self._departureStart is not None:
                if self._departureHelped:
                    self.recoveriesHelped += 1
                else:
                    self.recoveryTimes.append(self.elapsed - self._departureStart)
                self._departureStart = None
                self._departureHelped = False
        self._wasOnRoad = onRoad

        self._sampleStopSigns()
        self._samplePedestrians()

    def _sampleStopSigns(self):
        """Track each sign from 150 units out, so we can separate a detection-range
        failure from a policy failure.

        Physics: full brake decelerates at MAX_BRAKE = 35 units/s^2, and the stopping
        zone is ~13 units, so entering it above sqrt(2*35*13) = 30.2 mph makes a full
        stop impossible no matter what the model does. If brakeOnsetDist is short and
        speedAtZoneEntry is above that, the model is reacting too late (detection). If
        it brakes early and still arrives fast, it is under-braking (policy).
        """
        car = self.worldGen.car
        speed = abs(car.returnVelocity())
        braking = car.breakPedal > 0.1

        for chunk in list(self.worldGen.worldObjects.values()):
            for obj in list(chunk):
                if not isinstance(obj, Autopilot_sim.StopSign):
                    continue

                dist = math.dist((car.position.x, car.position.z),
                                 (obj.world_position.x, obj.world_position.z))
                if dist > 150.0 and id(obj) not in self.stopSigns:
                    continue

                key = id(obj)
                isNew = key not in self.stopSigns
                rec = self.stopSigns.setdefault(key, {
                    "minSpeed": speed, "stopped": False, "resumed": False,
                    "framesInZone": 0, "minDist": dist, "brakeOnsetDist": None,
                    "speedAtBrakeOnset": None, "speedAtZoneEntry": None,
                    "speedAt40Units": None, "everInZone": False,
                    # If we were already braking when the sign came into range, any
                    # "onset" we record is some earlier cause (a pedestrian, a curve),
                    # not a reaction to this sign. Excluded rather than reported as a
                    # 150-unit detection, which is what the first version did.
                    "brakingOnArrival": braking,
                    "profile": {},
                    "restDist": None,
                    "peakSalience": 0,
                    "curvature": self.geometry.curvatureAt(obj.world_position),
                    "helped": False,
                    "stoppedAt": None,
                })
                if isNew and braking:
                    rec["brakingOnArrival"] = True

                approaching = dist < rec["minDist"]
                rec["minDist"] = min(rec["minDist"], dist)

                if approaching and dist < 150.0:
                    rec["peakSalience"] = max(rec["peakSalience"],
                                              getattr(self.worldGen, "last_sign_salience", 0))

                if approaching:
                    for bucket in APPROACH_BUCKETS:
                        if dist <= bucket and bucket not in rec["profile"]:
                            rec["profile"][bucket] = speed

                # where it actually came to rest, whether or not that registered
                if speed < 0.5 and rec["restDist"] is None and dist < 60.0:
                    rec["restDist"] = dist

                if approaching and braking and not rec["brakingOnArrival"] \
                        and rec["brakeOnsetDist"] is None:
                    rec["brakeOnsetDist"] = dist
                    rec["speedAtBrakeOnset"] = speed
                if approaching and rec["speedAt40Units"] is None and dist <= 40.0:
                    rec["speedAt40Units"] = speed

                trigger = getattr(obj, "trigger", None)
                inZone = False
                if trigger is not None:
                    try:
                        inZone = car.intersects(trigger).hit
                    except Exception:
                        inZone = False

                if inZone:
                    rec["everInZone"] = True
                    rec["framesInZone"] += 1
                    rec["minSpeed"] = min(rec["minSpeed"], speed)
                    if rec["speedAtZoneEntry"] is None:
                        rec["speedAtZoneEntry"] = speed
                    # has_triggered is set by WorldGenerator only on an exact full stop
                    if getattr(obj, "has_triggered", False):
                        if not rec["stopped"]:
                            rec["stoppedAt"] = self.elapsed
                        rec["stopped"] = True
                # A push only counts as "it would not resume" once the sign has actually
                # gone green -- WorldGenerator calls swapToGreen on a 1.5s delay, so
                # anything before that is you nudging a short-stopped car into the zone,
                # which is a placement failure, not a resume failure. Counting both as
                # the same thing is what made these indistinguishable.
                GREEN_DELAY = 1.5
                if (rec["stopped"] and not rec["resumed"] and speed < 5.0
                        and rec["stoppedAt"] is not None
                        and self.elapsed - rec["stoppedAt"] > GREEN_DELAY + 0.2
                        and any(held_keys[k] for k in ("w", "a", "s", "d"))):
                    rec["helped"] = True

                if not inZone and rec["everInZone"]:
                    if rec["stopped"] and speed > 5.0:
                        rec["resumed"] = True

    def _samplePedestrians(self):
        if self._pedestrianCooldown > 0:
            return
        car = self.worldGen.car
        for chunk in list(self.worldGen.worldObjects.values()):
            for obj in list(chunk):
                if not isinstance(obj, Autopilot_sim.Pedestrian):
                    continue
                d = math.dist((car.position.x, car.position.z), (obj.position.x, obj.position.z))
                if d < 2.0:
                    self.pedestrianStrikes += 1
                    self._pedestrianCooldown = 2.0
                    return

    # ------------------------------------------------------------------- results

    def finish(self):
        if self.finished:
            return
        self.finished = True
        if self.onFinish:
            self.onFinish(self.results())

    def results(self):
        n = max(1, self.totalSamples)
        dev = np.array(self.laneDeviations) if self.laneDeviations else np.array([0.0])
        speeds = np.array(self.speeds) if self.speeds else np.array([0.0])
        limits = np.array(self.speedLimits) if self.speedLimits else np.array([1.0])
        wheels = np.array(self.wheelAngles) if self.wheelAngles else np.array([0.0])
        heading = np.array(self.headingErrors) if self.headingErrors else np.array([0.0])

        # mean absolute wheel change per second -- high means a twitchy controller
        if len(wheels) > 1:
            smoothness = float(np.abs(np.diff(wheels)).mean()) / self.samplePeriod
        else:
            smoothness = 0.0

        approached = [r for r in self.stopSigns.values() if r["everInZone"]]
        encountered = len(approached)
        stopped = sum(1 for r in approached if r["stopped"])
        nearlyStopped = sum(1 for r in approached if r["minSpeed"] < 2.0)
        resumed = sum(1 for r in approached if r["resumed"])
        resumedUnaided = sum(1 for r in approached if r["resumed"] and not r["helped"])
        stoppedNeededHelp = sum(1 for r in approached if r["stopped"] and r["helped"])

        if self._departureStart is not None:
            self.neverRecovered = 1   # still off the road when the run ended

        def med(vals):
            vals = [v for v in vals if v is not None]
            return float(np.median(vals)) if vals else None

        brakeOnset = med([r["brakeOnsetDist"] for r in approached])
        brakeOnsetSpeed = med([r["speedAtBrakeOnset"] for r in approached])
        zoneEntry = med([r["speedAtZoneEntry"] for r in approached])
        speed40 = med([r["speedAt40Units"] for r in approached])
        stoppableLimit = math.sqrt(2 * 35.0 * 13.0)   # 30.2 mph
        entrySpeeds = [r["speedAtZoneEntry"] for r in approached if r["speedAtZoneEntry"] is not None]
        arrivedTooFast = sum(1 for s in entrySpeeds if s > stoppableLimit)
        brakingOnArrival = sum(1 for r in approached if r["brakingOnArrival"])

        restDists = [r["restDist"] for r in approached if r["restDist"] is not None]
        stoppedInZone = sum(1 for d in restDists if STOP_ZONE_NEAR <= d <= STOP_ZONE_FAR)
        stoppedPastZone = sum(1 for d in restDists if d < STOP_ZONE_NEAR)
        stoppedShort = sum(1 for d in restDists if d > STOP_ZONE_FAR)

        # Split every approached sign by outcome, so a miss can be attributed.
        reacted = [r for r in approached if r["restDist"] is not None]
        ignored = [r for r in approached if r["restDist"] is None]
        missDiag = {
            "stoppedPeakSalience": med([r["peakSalience"] for r in reacted]),
            "ignoredPeakSalience": med([r["peakSalience"] for r in ignored]),
            "stoppedCurvature": med([r["curvature"] for r in reacted]),
            "ignoredCurvature": med([r["curvature"] for r in ignored]),
            "ignoredNeverVisible": sum(1 for r in ignored if r["peakSalience"] < 100),
        }

        approachProfile = {}
        for bucket in APPROACH_BUCKETS:
            vals = [r["profile"][bucket] for r in approached if bucket in r["profile"]]
            approachProfile[bucket] = float(np.median(vals)) if vals else None

        return {
            "duration": self.elapsed,
            "distance": self.distance,
            "laneDevMean": float(dev.mean()),
            # median is the honest central figure: a single off-road excursion drags the
            # mean above the p95, which is exactly what happened on the first 240s runs
            "laneDevMedian": float(np.median(dev)),
            "laneDevP95": float(np.percentile(dev, 95)),
            "laneDevMax": float(dev.max()),
            "headingErrMean": float(heading.mean()),
            "timeOnRoadPct": 100.0 * (n - self.offRoadSamples) / n,
            "timeCorrectLanePct": 100.0 * (n - self.offRoadSamples - self.wrongLaneSamples) / n,
            "departures": self.departures,
            "departuresPerMin": self.departures / max(1e-6, self.elapsed / 60.0),
            "distanceBeforeFirstDeparture": self.distanceBeforeFirstDeparture,
            "speedMean": float(speeds.mean()),
            "speedLimitMean": float(limits.mean()),
            "speedRatio": float((speeds / np.maximum(limits, 1.0)).mean()),
            "steerSmoothness": smoothness,
            "signsEncountered": encountered,
            "signsFullyStopped": stopped,
            "signsNearlyStopped": nearlyStopped,
            "signsResumed": resumed,
            "signsResumedUnaided": resumedUnaided,
            "stoppedNeededHelp": stoppedNeededHelp,
            "recoveriesUnaided": len(self.recoveryTimes),
            "recoveriesHelped": self.recoveriesHelped,
            "neverRecovered": self.neverRecovered,
            "recoveryTimeMedian": med(self.recoveryTimes),
            "recoveryTimeMax": float(max(self.recoveryTimes)) if self.recoveryTimes else None,
            "brakeOnsetDist": brakeOnset,
            "brakeOnsetSpeed": brakeOnsetSpeed,
            "speedAt40Units": speed40,
            "speedAtZoneEntry": zoneEntry,
            "arrivedTooFast": arrivedTooFast,
            "brakingOnArrival": brakingOnArrival,
            "approachProfile": approachProfile,
            **missDiag,
            "cameToRest": len(restDists),
            "restDistMedian": med(restDists),
            "stoppedInZone": stoppedInZone,
            "stoppedPastZone": stoppedPastZone,
            "stoppedShort": stoppedShort,
            "stoppableLimit": stoppableLimit,
            "pedestrianStrikes": self.pedestrianStrikes,
            "interventionPct": 100.0 * self.interventionSamples / n,
        }


def _fmt(v, unit=""):
    return "n/a" if v is None else f"{v:.1f}{unit}"


def printResults(modelName, r):
    line = "=" * 66
    print(f"\n{line}\nEVALUATION  {modelName}\n{line}")
    print(f"  ran {r['duration']:.0f}s, travelled {r['distance']:.0f} units "
          f"(world seed {EVAL_WORLD_SEED})")
    print("\n  LANE KEEPING")
    print(f"    median deviation from lane centre: {r.get('laneDevMedian', float('nan')):.2f} units")
    print(f"    mean deviation                  : {r['laneDevMean']:.2f} units")
    print(f"    p95 / max deviation             : {r['laneDevP95']:.2f} / {r['laneDevMax']:.2f}")
    print(f"    mean heading error              : {r['headingErrMean']:.1f} deg")
    print(f"    time on road                    : {r['timeOnRoadPct']:.1f}%")
    print(f"    time in correct lane            : {r['timeCorrectLanePct']:.1f}%")
    print(f"    departures                      : {r['departures']} ({r['departuresPerMin']:.2f}/min)")
    print(f"      recovered on its own          : {r.get('recoveriesUnaided', 0)}"
          f"   (median {_fmt(r.get('recoveryTimeMedian'), ' s')}, worst {_fmt(r.get('recoveryTimeMax'), ' s')})")
    print(f"      needed you to rescue it       : {r.get('recoveriesHelped', 0)}")
    print(f"      still off road at the end     : {r.get('neverRecovered', 0)}")
    if r["distanceBeforeFirstDeparture"] is not None:
        print(f"    distance before 1st departure   : {r['distanceBeforeFirstDeparture']:.0f} units")
    else:
        print(f"    distance before 1st departure   : never departed")
    print(f"    steering smoothness             : {r['steerSmoothness']:.0f} deg/s of wheel movement")
    print("\n  SPEED")
    print(f"    mean speed / mean limit         : {r['speedMean']:.1f} / {r['speedLimitMean']:.1f} mph")
    print(f"    speed as fraction of limit      : {r['speedRatio'] * 100:.0f}%")
    print("\n  STOP SIGNS")
    print(f"    encountered                     : {r['signsEncountered']}")
    print(f"    fully stopped (sign turned green): {r['signsFullyStopped']}")
    print(f"    got below 2 mph                 : {r['signsNearlyStopped']}")
    print(f"    resumed after green             : {r['signsResumed']}"
          f"   of which UNAIDED: {r.get('signsResumedUnaided', r['signsResumed'])}")
    print(f"    stops that needed a push        : {r.get('stoppedNeededHelp', 0)}")
    encountered = r["signsEncountered"]
    rested = r.get("cameToRest", 0)
    print(f"    NEVER stopped at all            : {encountered - rested} of {encountered}"
          f"   <- drove straight through")
    print(f"    came to a rest near a sign      : {rested} "
          f"(median {_fmt(r.get('restDistMedian'), ' units from sign')})")
    print(f"      of those, INSIDE the zone     : {r.get('stoppedInZone', 0)}   "
          f"(valid band is {STOP_ZONE_NEAR:.1f}-{STOP_ZONE_FAR:.1f} units)")
    print(f"      stopped SHORT of the zone     : {r.get('stoppedShort', 0)}   "
          f"<- needs a nudge; the sign never turns green")
    print(f"      rolled PAST the zone          : {r.get('stoppedPastZone', 0)}")

    ign = encountered - rested
    if ign:
        print(f"\n  WHY THE MISSES?  (never saw it, or saw it and ignored it)")
        print(f"    peak sign visibility, reacted : {_fmt(r.get('stoppedPeakSalience'), ' red px')}")
        print(f"    peak sign visibility, ignored : {_fmt(r.get('ignoredPeakSalience'), ' red px')}")
        print(f"    road curvature, reacted       : {_fmt(r.get('stoppedCurvature'), ' deg')}")
        print(f"    road curvature, ignored       : {_fmt(r.get('ignoredCurvature'), ' deg')}")
        print(f"    ignored signs never clearly visible: {r.get('ignoredNeverVisible', 0)} of {ign}")
        nv = r.get('ignoredNeverVisible', 0)
        if nv >= max(1, ign * 0.6):
            print("    => it NEVER SAW them -> perception/geometry, not policy")
        else:
            print("    => it SAW them and drove on -> policy, not perception")

    def fmt(v, unit=""):
        return "n/a" if v is None else f"{v:.1f}{unit}"

    print("\n  STOP APPROACH  (is it a detection problem or a policy problem?)")
    profile = r.get("approachProfile") or {}
    if profile:
        cols = [b for b in APPROACH_BUCKETS if profile.get(b) is not None]
        if cols:
            print("    median speed vs distance to the sign:")
            print("      units out  " + "".join(f"{b:>7}" for b in cols))
            print("      mph        " + "".join(f"{profile[b]:>7.1f}" for b in cols))
            print("      (flat until ~20 = late detection; declining from ~100 = it saw it)")
    print(f"    already braking on arrival      : {r.get('brakingOnArrival', 0)} of "
          f"{r['signsEncountered']}")
    print(f"    brake first applied at          : {fmt(r['brakeOnsetDist'], ' units out')}"
          f"   (unreliable when the model brakes often)")
    print(f"    speed at that moment            : {fmt(r['brakeOnsetSpeed'], ' mph')}")
    print(f"    speed 40 units out              : {fmt(r['speedAt40Units'], ' mph')}")
    print(f"    speed entering stopping zone    : {fmt(r['speedAtZoneEntry'], ' mph')}")
    print(f"    arrived above the {r['stoppableLimit']:.1f} mph limit : "
          f"{r['arrivedTooFast']} of {r['signsEncountered']}  (a full stop is physically")
    print(f"                                      impossible above that speed)")
    if r["brakeOnsetDist"] is not None:
        if r["brakeOnsetDist"] < 30:
            print("    => brakes LATE: reacting only once the sign is large -> detection range")
        elif r["speedAtZoneEntry"] is not None and r["speedAtZoneEntry"] > r["stoppableLimit"]:
            print("    => brakes EARLY but still arrives too fast -> under-braking, a policy issue")
        else:
            print("    => approach profile looks healthy")
    print("\n  OBSTACLES")
    print(f"    pedestrian near-misses (<2 units): {r['pedestrianStrikes']}")
    if r.get("interventionPct", 0) > 0.05:
        print(f"\n  !! manual input during {r['interventionPct']:.1f}% of the run -- lane and")
        print(f"     speed figures include whatever you were doing at the time")
    print(line + "\n")


def writeResults(modelName, r, signDensity=1):
    out = Path(__file__).parent / "Evaluations"
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out / f"eval_{stamp}.txt"
    with open(path, "w") as f:
        f.write(f"model: {modelName}\nseed: {EVAL_WORLD_SEED}\n")
        f.write(f"signDensity: {signDensity}\n")
        for k, v in r.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved evaluation to {path}")
    return path


# =============================================================================== UI


class EvaluationHUD(Entity):
    """Live readout while the run is in progress."""

    def __init__(self, run, modelName, queueLabel=None, **kwargs):
        kwargs.setdefault("parent", camera.ui)
        super().__init__(**kwargs)
        self.run = run
        if queueLabel:
            Text(parent=self, text=f"model {queueLabel}", position=(0.39, 0.455),
                 scale=0.8, color=color.yellow)

        # Top-RIGHT: the car's own Speedometer / SpeedLimitText / statusCircle occupy
        # the top-left corner (x ~ -0.82, y 0.31..0.45) and are parented to camera.ui.
        self.bg = Entity(parent=self, model="quad", color=color.rgba32(15, 15, 25, 205),
                         scale=(0.46, 0.30), position=(0.60, 0.30), z=1)
        self.title = Text(parent=self, text="EVALUATING", position=(0.39, 0.42),
                          scale=1.15, color=color.cyan)
        self.modelText = Text(parent=self, text=modelName[:34], position=(0.39, 0.385),
                              scale=0.75, color=color.light_gray)
        self.body = Text(parent=self, text="", position=(0.39, 0.345), scale=0.85,
                         color=color.white)
        self.barBack = Entity(parent=self, model="quad", color=color.dark_gray,
                              scale=(0.42, 0.012), position=(0.60, 0.175), z=0)
        self.barFill = Entity(parent=self, model="quad", color=color.cyan,
                              scale=(0.001, 0.012), position=(0.39, 0.175), z=-1,
                              origin=(-0.5, 0))

    def update(self):
        r = self.run
        if r.finished:
            return
        frac = min(1.0, r.elapsed / r.duration)
        self.barFill.scale_x = max(0.001, 0.42 * frac)

        dev = np.mean(r.laneDeviations) if r.laneDeviations else 0.0
        onRoad = 100.0 * (r.totalSamples - r.offRoadSamples) / max(1, r.totalSamples)
        stopped = sum(1 for v in r.stopSigns.values() if v["stopped"])
        self.body.text = (
            f"time      {r.elapsed:5.1f} / {r.duration:.0f} s\n"
            f"distance  {r.distance:6.0f} units\n"
            f"lane dev  {dev:5.2f} units\n"
            f"on road   {onRoad:5.1f} %\n"
            f"departs   {r.departures}\n"
            f"stops     {stopped} / {len(r.stopSigns)}"
        )


class ResultsPanel(Entity):
    def __init__(self, modelName, results, onRerun=None, onClose=None, **kwargs):
        kwargs.setdefault("parent", camera.ui)
        kwargs.setdefault("z", -10)
        super().__init__(**kwargs)
        r = results

        Entity(parent=self, model="quad", color=color.rgba32(20, 20, 30, 240),
               scale=(1.18, 0.92), z=1)
        Text(parent=self, text="Evaluation Results", position=(-0.55, 0.40),
             scale=1.5, color=color.cyan)
        Text(parent=self, text=modelName[:52], position=(-0.55, 0.355),
             scale=0.85, color=color.light_gray)

        def verdictColor(ok, warn):
            return color.lime if ok else (color.yellow if warn else color.red)

        left = (
            f"LANE KEEPING\n"
            f"  median deviation    {r.get('laneDevMedian', float('nan')):.2f} units\n"
            f"  mean deviation      {r['laneDevMean']:.2f} units\n"
            f"  p95 deviation       {r['laneDevP95']:.2f} units\n"
            f"  max deviation       {r['laneDevMax']:.2f} units\n"
            f"  heading error       {r['headingErrMean']:.1f} deg\n"
            f"  time on road        {r['timeOnRoadPct']:.1f} %\n"
            f"  correct lane        {r['timeCorrectLanePct']:.1f} %\n"
            f"  departures          {r['departures']}  ({r['departuresPerMin']:.2f}/min)\n"
            f"  steer movement      {r['steerSmoothness']:.0f} deg/s\n"
        )
        right = (
            f"SPEED\n"
            f"  mean speed          {r['speedMean']:.1f} mph\n"
            f"  mean limit          {r['speedLimitMean']:.1f} mph\n"
            f"  fraction of limit   {r['speedRatio'] * 100:.0f} %\n\n"
            f"STOP SIGNS\n"
            f"  encountered         {r['signsEncountered']}\n"
            f"  fully stopped       {r['signsFullyStopped']}\n"
            f"  below 2 mph         {r['signsNearlyStopped']}\n"
            f"  resumed on green    {r['signsResumed']}\n\n"
            f"STOP APPROACH\n"
            f"  brakes at           {_fmt(r['brakeOnsetDist'], ' units out')}\n"
            f"  speed 40 units out  {_fmt(r['speedAt40Units'], ' mph')}\n"
            f"  speed entering zone {_fmt(r['speedAtZoneEntry'], ' mph')}\n"
            f"  too fast to stop    {r['arrivedTooFast']} / {r['signsEncountered']}"
            f"  (limit {r['stoppableLimit']:.0f} mph)\n\n"
            f"OBSTACLES\n"
            f"  pedestrian close    {r['pedestrianStrikes']}\n"
        )
        Text(parent=self, text=left, position=(-0.55, 0.30), scale=0.9, color=color.white)
        Text(parent=self, text=right, position=(0.03, 0.30), scale=0.9, color=color.white)

        headline = (f"on road {r['timeOnRoadPct']:.1f}%   |   "
                    f"lane dev {r['laneDevMean']:.2f}   |   "
                    f"stops {r['signsFullyStopped']}/{r['signsEncountered']}   |   "
                    f"{r['distance']:.0f} units in {r['duration']:.0f}s")
        Text(parent=self, text=headline, position=(-0.55, -0.24), scale=1.0,
             color=verdictColor(r["timeOnRoadPct"] > 97 and r["departures"] == 0,
                                r["timeOnRoadPct"] > 85))

        Text(parent=self, text=f"world seed {EVAL_WORLD_SEED} - every model drives this same course",
             position=(-0.55, -0.30), scale=0.75, color=color.gray)

        if onRerun:
            b = Button(text="Evaluate Another Model", parent=self, scale=(0.42, 0.075),
                       position=(-0.24, -0.39), color=color.azure, text_color=color.white)
            b.on_click = onRerun
        if onClose:
            b2 = Button(text="Done", parent=self, scale=(0.22, 0.075),
                        position=(0.22, -0.39), color=color.dark_gray, text_color=color.white)
            b2.on_click = onClose


class ComparisonPanel(Entity):
    """Side-by-side results after a queued batch.

    Reading several eval files by hand is where mistakes creep in -- it is how a
    density-1 run got compared against a density-3 baseline earlier. Everything here ran
    on the same course, same duration, same seed, in one sitting.
    """

    def __init__(self, results, signDensity, duration, onRerun=None, onClose=None, **kwargs):
        kwargs.setdefault("parent", camera.ui)
        kwargs.setdefault("z", -10)
        super().__init__(**kwargs)

        Entity(parent=self, model="quad", color=color.rgba32(20, 20, 30, 242),
               scale=(1.5, 0.94), z=1)
        Text(parent=self, text="Comparison", position=(-0.71, 0.42), scale=1.5, color=color.cyan)
        Text(parent=self, text=f"{len(results)} models - {duration:.0f}s each - "
                               f"sign density x{signDensity} - seed {EVAL_WORLD_SEED}",
             position=(-0.71, 0.375), scale=0.8, color=color.gray)

        rows = [
            ("on road %", lambda r: f"{r['timeOnRoadPct']:.1f}", True),
            ("correct lane %", lambda r: f"{r['timeCorrectLanePct']:.1f}", True),
            ("departures", lambda r: f"{r['departures']}", False),
            ("lane dev median", lambda r: f"{r.get('laneDevMedian', float('nan')):.2f}", False),
            ("lane dev p95", lambda r: f"{r['laneDevP95']:.2f}", False),
            ("heading err", lambda r: f"{r['headingErrMean']:.2f}", False),
            ("steer movement", lambda r: f"{r['steerSmoothness']:.0f}", False),
            ("speed / limit", lambda r: f"{r['speedRatio'] * 100:.0f}%", True),
            ("signs stopped", lambda r: f"{r['signsFullyStopped']} / {r['signsEncountered']}", True),
            ("stopped in zone", lambda r: f"{r.get('stoppedInZone', 0)}", True),
            ("resumed unaided", lambda r: f"{r.get('signsResumedUnaided', r['signsResumed'])}", True),
            ("recovered alone", lambda r: f"{r.get('recoveriesUnaided', 0)}", True),
            ("you rescued it", lambda r: f"{r.get('recoveriesHelped', 0)}", False),
            ("rest distance", lambda r: _fmt(r.get('restDistMedian')), False),
            ("pedestrian close", lambda r: f"{r['pedestrianStrikes']}", False),
            ("your input %", lambda r: f"{r.get('interventionPct', 0):.1f}", False),
        ]

        colX = [-0.30 + i * 0.34 for i in range(len(results))]
        for i, (name, _) in enumerate(results):
            Text(parent=self, text=name[:26], position=(colX[i], 0.30), scale=0.8,
                 color=color.white)

        for j, (label, fn, higherBetter) in enumerate(rows):
            y = 0.245 - j * 0.045
            Text(parent=self, text=label, position=(-0.71, y), scale=0.85,
                 color=color.light_gray)
            vals = []
            for _, r in results:
                try:
                    vals.append(fn(r))
                except Exception:
                    vals.append("-")
            # highlight the winner where the direction is unambiguous
            best = None
            try:
                nums = [float(v.rstrip('%').split('/')[0]) for v in vals]
                best = (max if higherBetter else min)(range(len(nums)), key=lambda k: nums[k])
            except Exception:
                best = None
            for i, v in enumerate(vals):
                Text(parent=self, text=v, position=(colX[i], y), scale=0.85,
                     color=color.lime if best == i and len(results) > 1 else color.white)

        Text(parent=self, text="green = best of the batch on that row, where 'better' is unambiguous",
             position=(-0.71, -0.36), scale=0.7, color=color.gray)

        if onRerun:
            b = Button(text="Run More", parent=self, scale=(0.3, 0.075), position=(-0.2, -0.42),
                       color=color.azure, text_color=color.white)
            b.on_click = onRerun
        if onClose:
            b = Button(text="Done", parent=self, scale=(0.2, 0.075), position=(0.18, -0.42),
                       color=color.dark_gray, text_color=color.white)
            b.on_click = onClose


class ModelPicker(Entity):
    """Paginated model chooser, styled to match the in-sim selector."""

    PAGE_SIZE = 5

    def __init__(self, onPick, onCancel=None, page=0, duration=90.0, signDensity=1, **kwargs):
        kwargs.setdefault("parent", camera.ui)
        kwargs.setdefault("z", -10)
        super().__init__(**kwargs)
        self.onPick = onPick
        self.onCancel = onCancel
        self.page = page
        self.duration = duration
        self.signDensity = signDensity
        self.selected = []          # ordered; models run back to back in this order
        self._rebuildQueued = False

        modelsDir = Path(__file__).parent / "Models"
        self.files = sorted(modelsDir.glob("*.pth"), key=lambda f: f.stat().st_mtime, reverse=True) \
            if modelsDir.exists() else []

        self._build()

    # Paging and duration changes rebuild the CONTENTS of this entity rather than
    # replacing the entity itself, and the rebuild is deferred to the next frame.
    # Destroying and recreating the panel inside input() meant the replacement widgets
    # received the very same scroll event and recursed.
    def _clearChildren(self):
        for child in list(self.children):
            Autopilot_sim.destroy_all(child)

    def _queueRebuild(self):
        self._rebuildQueued = True

    def update(self):
        if self._rebuildQueued:
            self._rebuildQueued = False
            self._clearChildren()
            self._build()

    def setPage(self, page):
        totalPages = max(1, math.ceil(len(self.files) / self.PAGE_SIZE)) if self.files else 1
        page = max(0, min(page, totalPages - 1))
        if page != self.page:
            self.page = page
            self._queueRebuild()

    def setDuration(self, duration):
        if abs(duration - self.duration) > 1e-6:
            self.duration = duration
            self._queueRebuild()

    def toggleModel(self, path):
        """Selection is a queue, not a radio button -- order is the order they run in."""
        if path in self.selected:
            self.selected.remove(path)
        else:
            self.selected.append(path)
        self._queueRebuild()

    def _build(self):
        onCancel = self.onCancel

        Entity(parent=self, model="quad", color=color.rgba32(20, 20, 30, 238),
               scale=(1.15, 0.92), z=1)

        if not self.files:
            Text(parent=self, text="No .pth models found in Models/", position=(-0.35, 0),
                 scale=1.2, color=color.red)
            if onCancel:
                b = Button(text="Back", parent=self, scale=(0.25, 0.08), position=(0, -0.2),
                           color=color.dark_gray, text_color=color.white)
                b.on_click = onCancel
            return

        totalPages = max(1, math.ceil(len(self.files) / self.PAGE_SIZE))
        self.page = max(0, min(self.page, totalPages - 1))

        Text(parent=self, text=f"Evaluate Models  (page {self.page + 1} of {totalPages})",
             position=(-0.55, 0.40), scale=1.4, color=color.cyan)
        Text(parent=self, text="click to queue several - they run back to back on identical courses",
             position=(-0.55, 0.355), scale=0.8, color=color.gray)

        start = self.page * self.PAGE_SIZE
        for i, path in enumerate(self.files[start:start + self.PAGE_SIZE]):
            queued = path in self.selected
            order = self.selected.index(path) + 1 if queued else None
            label = f"{order}.  {path.name}" if queued else path.name
            b = Button(text=label, parent=self, scale=(0.95, 0.07),
                       position=(0, 0.24 - i * 0.09),
                       color=color.azure if queued else color.dark_gray,
                       text_color=color.white)

            def handler(p=path):
                self.toggleModel(p)
            b.on_click = handler

        # --- settings rows -------------------------------------------------------
        # Two separate rows. These used to share one line, where the "Stop signs:" label
        # at x=0.05 sat underneath the 240s button (0.085-0.215) and the dense button ran
        # past the panel edge at 0.595 against a 0.575 boundary.
        Text(parent=self, text="Run length:", position=(-0.55, -0.185), scale=0.95,
             color=color.light_gray)
        for i, secs in enumerate((60, 90, 150, 240, 600)):
            selected = abs(self.duration - secs) < 1e-6
            b = Button(text=f"{secs}s", parent=self, scale=(0.125, 0.062),
                       position=(-0.24 + i * 0.135, -0.20),
                       color=color.azure if selected else color.dark_gray,
                       text_color=color.white)

            def pickDuration(s=secs):
                self.setDuration(float(s))
            b.on_click = pickDuration

        # "normal" keeps the course identical to earlier runs so old results stay
        # comparable; dense trades that for ~3x the signs, which is the only way to get a
        # usable sample of stop behaviour out of a single run.
        Text(parent=self, text="Stop signs:", position=(-0.55, -0.265), scale=0.95,
             color=color.light_gray)
        for i, (dens, lab) in enumerate(((1, "normal"), (3, "dense x3"))):
            selected = self.signDensity == dens
            b = Button(text=lab, parent=self, scale=(0.17, 0.062),
                       position=(-0.20 + i * 0.19, -0.28),
                       color=color.azure if selected else color.dark_gray,
                       text_color=color.white)

            def pickDensity(d=dens):
                if d != self.signDensity:
                    self.signDensity = d
                    self._queueRebuild()
            b.on_click = pickDensity

        mins = len(self.selected) * self.duration / 60.0
        Text(parent=self, text=f"stop signs are sparse - 240s+ and dense x3 give a usable sample"
                               + (f"     |     queue: {len(self.selected)} models, ~{mins:.0f} min"
                                  if self.selected else ""),
             position=(-0.55, -0.325), scale=0.7, color=color.gray)

        runLabel = (f"Run {len(self.selected)} model{'s' if len(self.selected) != 1 else ''}"
                    if self.selected else "Select a model above")
        b = Button(text=runLabel, parent=self, scale=(0.42, 0.075), position=(-0.20, -0.40),
                   color=color.lime if self.selected else color.dark_gray,
                   text_color=color.black if self.selected else color.gray)
        if self.selected:
            def runQueue():
                self.onPick(list(self.selected), self.duration, self.signDensity)
            b.on_click = runQueue

        if self.selected:
            b = Button(text="Clear", parent=self, scale=(0.16, 0.075), position=(0.12, -0.40),
                       color=color.dark_gray, text_color=color.white)

            def clearSel():
                self.selected = []
                self._queueRebuild()
            b.on_click = clearSel

        if onCancel:
            b = Button(text="Back", parent=self, scale=(0.16, 0.075), position=(0.40, -0.40),
                       color=color.dark_gray, text_color=color.white)
            b.on_click = onCancel

    def input(self, key):
        if not self.files or not self.enabled:
            return
        if key == "scroll up":
            self.setPage(self.page - 1)
        elif key == "scroll down":
            self.setPage(self.page + 1)


# ============================================================ orchestration


def startEvaluationFlow(makeWorld, onExit=None):
    """Open the picker, run the chosen model on a FRESH world, then show results.

    makeWorld() must reseed and build a brand new WorldGenerator. A world is torn down
    and rebuilt before every run: the track scrolls, obstacles get consumed and the
    speed limit changes as a run proceeds, so reusing one would hand the second model a
    different course and silently void the comparison.
    """
    state = {"world": None, "hud": None, "run": None, "panel": None, "picker": None}

    def destroyUI():
        # destroy_all, not destroy: ursina's destroy() does not cascade to children, so
        # a plain destroy() removes the container and leaves every button and label
        # sitting on screen.
        for key in ("hud", "run", "panel", "picker"):
            if state[key]:
                Autopilot_sim.destroy_all(state[key])
                state[key] = None

    def teardownWorld():
        if state["world"] is not None:
            try:
                state["world"].teardown()
            except Exception as e:
                print(f"EVALUATION: world teardown issue ({e})")
            state["world"] = None

    def leave():
        destroyUI()
        teardownWorld()
        if onExit:
            onExit()

    def defer(fn):
        """Run fn a frame later.

        Tearing down and rebuilding the UI tree from inside a click handler lets the
        freshly created widgets receive the very same mouse event -- clicking "Evaluate
        Another Model" would otherwise hand the mouse-up straight to whichever model
        button spawned underneath the cursor.
        """
        invoke(fn, delay=0.05)

    def openPicker(duration=90.0, signDensity=1):
        destroyUI()
        state["picker"] = ModelPicker(
            onPick=lambda p, d, s: defer(lambda: begin(p, d, s)),
            onCancel=lambda: defer(leave),
            duration=duration,
            signDensity=signDensity,
        )

    def begin(modelPaths, duration, signDensity=1):
        """Queue one or more models. Each gets a freshly rebuilt world, so they all face
        an identical course and the comparison at the end is like for like."""
        if not isinstance(modelPaths, (list, tuple)):
            modelPaths = [modelPaths]
        state["queue"] = list(modelPaths)
        state["results"] = []
        state["duration"] = duration
        state["density"] = signDensity
        state["total"] = len(modelPaths)
        runNext()

    def runNext():
        destroyUI()
        teardownWorld()

        if not state["queue"]:
            if state["results"]:
                state["panel"] = ComparisonPanel(
                    state["results"], state["density"], state["duration"],
                    onRerun=lambda: defer(lambda: openPicker(state["duration"], state["density"])),
                    onClose=lambda: defer(leave))
            else:
                openPicker(state["duration"], state["density"])
            return

        modelPath = state["queue"].pop(0)
        duration = state["duration"]
        signDensity = state["density"]
        index = state["total"] - len(state["queue"])

        world = makeWorld(signDensity)
        state["world"] = world

        if not world.load_specified_model(modelPath):
            print(f"EVALUATION: {modelPath.name} failed to load, skipping.")
            teardownWorld()
            defer(runNext)
            return

        # Start stationary, in the driving lane rather than straddling the centreline
        world.car.velocity = 0.0
        world.car.wheelAngle = 0.0
        world.car.accelerationPedal = 0.0
        world.car.breakPedal = 0.0
        try:
            geometry = LaneGeometry(world)
            world.car.position += geometry.laneCentreOffsetVector(world.car.position)
        except Exception as e:
            print(f"EVALUATION: could not align car to lane ({e})")
        world.autopilot_enabled = True
        # Lock out anything that would alter a scored run: training, recording toggles,
        # model swaps, weather, the pause panel. Camera views stay available.
        world.evaluationLocked = True

        print(f"\nEVALUATION [{index}/{state['total']}]: driving {modelPath.name} for "
              f"{duration:.0f}s on seeded course {EVAL_WORLD_SEED} (sign density x{signDensity})...")

        def done(results):
            world.autopilot_enabled = False
            world.evaluationLocked = False
            world.car.accelerationPedal = 0.0
            world.car.breakPedal = 0.0
            printResults(modelPath.name, results)
            writeResults(modelPath.name, results, signDensity)
            state["results"].append((modelPath.name, results))

            if state["queue"]:
                # straight on to the next model; the comparison comes at the end
                defer(runNext)
                return

            if state["hud"]:
                Autopilot_sim.destroy_all(state["hud"])
                state["hud"] = None

            if len(state["results"]) > 1:
                state["panel"] = ComparisonPanel(
                    state["results"], signDensity, duration,
                    onRerun=lambda: defer(lambda: openPicker(duration, signDensity)),
                    onClose=lambda: defer(leave))
            else:
                state["panel"] = ResultsPanel(
                    modelPath.name, results,
                    onRerun=lambda: defer(lambda: openPicker(duration, signDensity)),
                    onClose=lambda: defer(leave),
                )

        state["run"] = EvaluationRun(world, duration=duration, onFinish=done)
        state["hud"] = EvaluationHUD(state["run"], modelPath.name,
                                     queueLabel=f"{index} of {state['total']}")

    openPicker()
