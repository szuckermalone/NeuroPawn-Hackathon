"""Face Mesh landmarks and time-weighted eye closure metrics."""
import collections
import numpy as np
LEFT=(33,160,158,133,153,144)
RIGHT=(362,385,387,263,373,380)
MOUTH=(13,14,61,291)

def ear(pts, ids, width, height):
    a,b,c,d,e,f=[np.array((pts[i].x*width,pts[i].y*height)) for i in ids]
    return float((np.linalg.norm(b-f)+np.linalg.norm(c-e))/(2*max(np.linalg.norm(a-d),1e-6)))

class EyeMonitor:
    def __init__(self, threshold=.20, window=60):
        self.threshold=threshold; self.window=window
        self.observations=collections.deque();self.blinks=collections.deque()
        self.was_closed=False;self.closed_since=None;self.last_observed=None

    def update(self, now, landmarks, width, height):
        # Missing face is explicitly unknown, never counted as open eyes.
        if landmarks is None:
            self.was_closed=False;self.closed_since=None;self.last_observed=None
            return None
        eye_ratio=(ear(landmarks,LEFT,width,height)+ear(landmarks,RIGHT,width,height))/2
        closed=eye_ratio<self.threshold
        top,bottom,left,right=[np.array((landmarks[i].x*width,landmarks[i].y*height)) for i in MOUTH]
        mar=float(np.linalg.norm(top-bottom)/max(np.linalg.norm(left-right),1e-6))
        if closed and not self.was_closed:self.closed_since=now
        if not closed and self.was_closed and self.closed_since is not None:
            duration=now-self.closed_since
            if .08<=duration<=.7:self.blinks.append(now)
            self.closed_since=None
        self.was_closed=closed
        # Weight by elapsed observed time to avoid FPS-dependent PERCLOS.
        if self.last_observed is not None:
            dt=max(0.,min(now-self.last_observed,.5))
            if dt:self.observations.append((now,dt,float(closed)))
        self.last_observed=now
        while self.observations and now-self.observations[0][0]>self.window:
            self.observations.popleft()
        while self.blinks and now-self.blinks[0]>60:self.blinks.popleft()
        observed=sum(dt for _,dt,_ in self.observations)
        perclos=sum(dt*val for _,dt,val in self.observations)/observed if observed>=10. else None
        return dict(eye_closed=float(closed),perclos=perclos,
                    blink_rate=float(len(self.blinks)),
                    closure_seconds=float(now-self.closed_since) if closed and self.closed_since else 0.,
                    mouth_open=float(mar>.55),ear=eye_ratio,observed_seconds=observed)
