"""Live camera + Knight monitoring. Experimental use on a stationary simulator only.

Run: python -m multimodal.live --eeg none --camera 0
     python -m multimodal.live --eeg knight --port /dev/cu.usbmodem... --camera 0
"""
import argparse
import collections
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from .features import EEG_FEATURES, CAMERA_FEATURES, ALL_FEATURES, MONTAGE, eeg_features
from .camera_features import EyeMonitor
from .alerts import AlertConfig, AlertDispatcher


class EEGReader:
    def __init__(self, mode, port='', imu=False):
        self.mode=mode
        self.source=None
        self.buffer=collections.deque(maxlen=2000)
        self.fs=125
        if mode != 'none':
            # Reuse NeuroGuard's existing Knight implementation: starts channels and RLD.
            from acquisition import Source
            self.source=Source('synthetic' if mode=='synthetic' else 'live',port,imu,seed=123,demo_fatigue=False)
            self.fs=self.source.fs
            self.buffer=collections.deque(maxlen=int(self.fs*10))
        self.last_reason='No EEG source'
        self.last_sample_at=None
    def update(self):
        if not self.source:
            return None
        rows=self.source.read()
        if len(rows):
            self.buffer.extend(rows)
            self.last_sample_at=time.monotonic()
        else:
            self.last_reason='No new EEG samples'
            return None
        if len(self.buffer)<int(7.6*self.fs):
            self.last_reason='Collecting EEG window';return None
        recent=np.asarray(self.buffer)
        # Each row has a host/BrainFlow timestamp followed by the eight EEG channels in µV.
        features,reason=eeg_features(recent[-int(8*self.fs):,1:].T,self.fs)
        self.last_reason=reason
        return features
    def close(self):
        if self.source:self.source.close()


def main():
    ap=argparse.ArgumentParser(description='Experimental NeuroGuard multimodal research monitor')
    ap.add_argument('--eeg',choices=('none','synthetic','knight'),default='none')
    ap.add_argument('--port',default='',help='macOS Knight device /dev/cu.*')
    ap.add_argument('--imu',action='store_true',help='Select Knight IMU variant')
    ap.add_argument('--camera',default='0',help='OpenCV camera index, video path, or supported stream')
    ap.add_argument('--ear-threshold',type=float,default=.20)
    ap.add_argument('--model',default='',help='Trusted, independently trained model .joblib')
    ap.add_argument('--record',default='runs/multimodal/session.csv')
    ap.add_argument('--speak-alerts', action='store_true', help='Speak warnings through Mac selected audio output')
    ap.add_argument('--sms-alerts', action='store_true', help='Opt in to sending real SMS; requires Twilio environment credentials')
    ap.add_argument('--alert-phone', default=os.environ.get('NEUROGUARD_ALERT_PHONE',''), help='Recipient E.164 number')
    ap.add_argument('--alert-cooldown', type=float, default=300.0, help='Minimum seconds between alerts')
    args=ap.parse_args()
    if args.eeg=='knight' and not args.port:ap.error('--port is required for Knight mode')
    if args.eeg=='synthetic' and args.model:ap.error('Do not run trained model on synthetic EEG')
    if args.sms_alerts and (args.eeg != 'knight' or not args.model):
        ap.error('SMS is allowed only with Knight EEG and an explicitly supplied trained model')
    if args.sms_alerts and not args.alert_phone:
        ap.error('--alert-phone or NEUROGUARD_ALERT_PHONE is required with --sms-alerts')
    import cv2
    import mediapipe as mp
    try:
        mesh=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,
                                               min_detection_confidence=.5,min_tracking_confidence=.5)
    except AttributeError as ex:
        raise RuntimeError('Requires mediapipe==0.10.21 and Python 3.11') from ex
    cam=int(args.camera) if args.camera.isdecimal() else args.camera
    cap=cv2.VideoCapture(cam)
    if not cap.isOpened():
        mesh.close();raise RuntimeError('Cannot open camera; grant VS Code/Terminal camera permission on macOS')
    reader=None
    model=None; model_source='none'
    if args.model:
        import joblib
        model=joblib.load(args.model)  # Only load files produced by your team: pickle executes code.
        if list(getattr(model,'feature_names_in_',[]))!=list(ALL_FEATURES):
            raise ValueError('Model features do not match the current NeuroGuard feature schema')
        model_source='trained'
    monitor=EyeMonitor(threshold=args.ear_threshold)
    target=Path(args.record);target.parent.mkdir(parents=True,exist_ok=True)
    f=target.open('w',newline='');writer=csv.DictWriter(f,fieldnames=['timestamp','participant_id','session_id','camera_valid','eeg_valid','eeg_quality','source','risk','alert_state']+list(ALL_FEATURES));writer.writeheader()
    prev_log=0.;last_eeg_t=0.;last_eeg=None;warning_streak=0;last_alert=''
    dispatcher = AlertDispatcher(AlertConfig(sms_enabled=args.sms_alerts,
            speech_enabled=args.speak_alerts, recipient=args.alert_phone,
            cooldown_seconds=args.alert_cooldown))
    try:
        reader=EEGReader(args.eeg,args.port,args.imu)
        while True:
            ok,frame=cap.read()
            if not ok:break
            now=time.monotonic();h,w=frame.shape[:2]
            out=mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            landmarks=out.multi_face_landmarks[0].landmark if out.multi_face_landmarks else None
            cam_features=monitor.update(now,landmarks,w,h)
            fresh_eeg_window = now-last_eeg_t>=1
            if fresh_eeg_window:
                last_eeg=reader.update();last_eeg_t=now
            eeg_valid=last_eeg is not None
            camera_valid=cam_features is not None and cam_features['perclos'] is not None
            prediction=None
            if model is not None and eeg_valid and camera_valid:
                import pandas as pd
                inputs={**last_eeg,**{k:cam_features[k] for k in CAMERA_FEATURES}}
                prediction=float(model.predict_proba(pd.DataFrame([inputs],columns=list(ALL_FEATURES)))[0,1])
            # Do not represent absence of face or EEG as 'alert'. Camera-only fallback
            # is for demonstration, not an EEG classifier.
            if not cam_features:state='CAMERA UNAVAILABLE'
            elif not camera_valid:state='CALIBRATING CAMERA'
            elif args.eeg!='none' and not eeg_valid:state='EEG UNAVAILABLE'
            elif model is None:state='FEATURE MONITORING / UNTRAINED'
            else:
                # Persistence must count independent ~1-second EEG windows,
                # not individual 30-FPS camera frames.
                if fresh_eeg_window:
                    warning_streak=warning_streak+1 if prediction>=.65 else 0
                state='EXPERIMENTAL WARNING' if warning_streak>=3 else 'MONITORING'
            if state not in ('MONITORING','EXPERIMENTAL WARNING'):
                warning_streak=0
            if state=='EXPERIMENTAL WARNING' and last_alert!=state:
                print('\aEXPERIMENTAL WARNING: stop safely; monitoring is not validated')
            dispatcher.update(state, genuine_hardware=(args.eeg=='knight'),
                              validated_inputs=(eeg_valid and camera_valid and prediction is not None))
            dispatcher.check_errors()
            last_alert=state
            vals={k:float('nan') for k in ALL_FEATURES}
            if eeg_valid:vals.update(last_eeg)
            if cam_features:vals.update({k:cam_features[k] if cam_features[k] is not None else float('nan') for k in CAMERA_FEATURES})
            if now-prev_log>=1:
                writer.writerow({'timestamp':time.time(),'participant_id':'UNSET','session_id':target.stem,
                  'camera_valid':int(camera_valid),'eeg_valid':int(eeg_valid),
                  'eeg_quality':reader.last_reason,'source':args.eeg,
                  'risk':prediction if prediction is not None else '', 'alert_state':state,**vals})
                f.flush();prev_log=now
            lines=['NEUROGUARD: '+state,
                   f"EEG: {args.eeg} ({reader.last_reason})", 
                   'Theta/Alpha: '+(f"{vals['theta_alpha']:.2f}" if eeg_valid else 'unknown'),
                   'PERCLOS: '+(f"{cam_features['perclos']:.2f}" if camera_valid else 'unknown'),
                   'RF output: '+(f'{prediction:.2f}' if prediction is not None else 'NO TRAINED MODEL'),
                   'Press Q to quit']
            for i,label in enumerate(lines):
                cv2.putText(frame,label,(12,30+30*i),cv2.FONT_HERSHEY_SIMPLEX,.58,
                            (0,0,255) if state=='EXPERIMENTAL WARNING' and i==0 else (230,230,230),2)
            cv2.imshow('NeuroGuard experimental EEG + eyes',frame)
            if cv2.waitKey(1)&255==ord('q'):break
    finally:
        dispatcher.close()
        if reader:reader.close()
        cap.release();mesh.close();cv2.destroyAllWindows();f.close()
        print('Recorded',str(target))

if __name__=='__main__':main()
