# NeuroGuard SMS and spoken alert extension

Research/simulator prototype only. Not validated to detect driver sleepiness or prevent crashes. Never use while driving on roads. A spoken warning should recommend stopping safely rather than attempting to keep someone driving.

## What it does

- When the **trained** multimodal EEG/camera classifier enters `EXPERIMENTAL WARNING` for at least 3 processing updates, `multimodal.live` can optionally speak the warning through the Mac and/or send one SMS using Twilio
- SMS requires physical Knight EEG mode and a supplied trained model. Synthetic EEG and camera-only mode cannot send automatic SMS
- Alerts occur only on the transition into warning, with a default 300-second cooldown; network calls execute off the camera thread
- No SMS credentials or phone numbers are stored in files. No secrets should be committed to Git
- Invalid/missing camera or EEG input should show an unavailable state; do not interpret missing measurements as safe driving

## 1. Optional local spoken warning (Mac)

In your existing Python 3.11 environment, run:

```bash
python -m multimodal.alerts --test-speech
```

Audio goes to the Mac's currently selected output device. If paired with a Bluetooth speaker/car audio system, select that device as Mac audio output in System Settings > Sound > Output. This does not automatically route to a phone connected to the car.

## 2. Optional SMS delivery via Twilio

Create a Twilio account, obtain an SMS-enabled number, and check account verification, trial restrictions, sender registration, and regional messaging restrictions. SMS can incur costs and requires internet access. Twilio may accept a message without it reaching the phone.

Export credentials in the terminal session, NOT in source or a committed `.env` file:

```bash
export TWILIO_ACCOUNT_SID='ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
export TWILIO_AUTH_TOKEN='YOUR_SECRET_TOKEN'
export TWILIO_FROM_NUMBER='+1YOUR_TWILIO_NUMBER'
export NEUROGUARD_ALERT_PHONE='+1YOUR_PHONE_NUMBER'
```

Make one explicit test request **only to a consenting phone**:

```bash
python -m multimodal.alerts --send-test-sms
```

The provider SID printed means **accepted by provider**, not confirmed received/read aloud.

## 3. Real Knight + camera + classifier

Only when a model trained and independently evaluated on correctly paired, labelled EEG + camera data exists:

```bash
python -m multimodal.live \
  --eeg knight \
  --port /dev/cu.YOUR_KNIGHT_PORT \
  --camera 0 \
  --model models/neuroguard_multimodal_rf.joblib \
  --speak-alerts --sms-alerts \
  --alert-phone "$NEUROGUARD_ALERT_PHONE" \
  --alert-cooldown 300 \
  --record runs/multimodal/knight_alert_test.csv
```

Without a properly trained model, test the SMS and speech outputs independently rather than pretending the experimental EEG-change rules are validated sleepiness classifications.

## 4. Have your **phone** read an SMS through in-car audio

**iPhone with CarPlay**: Pair/connect iPhone with CarPlay and enable Siri's Announce Messages/Announce Notifications in iPhone settings or CarPlay. Test with a normal incoming SMS while parked. Plain Bluetooth pairing without CarPlay is not guaranteed to auto-announce texts. Some cars provide their own text-message reading integration.

**Android Auto**: Connect a compatible car/head unit and phone. Allow message notifications; new messages can be read aloud using the in-car notification or voice assistant. Actual automatic playback varies by car/device/settings. Test parked. Plain Bluetooth audio pairing alone does not guarantee spoken SMS.

**Alternative without CarPlay/Android Auto**: The `--speak-alerts` option can speak locally through a Mac selected Bluetooth audio output, which avoids dependence on the phone messaging announcement pathway. This requires the Mac to be the device connected to the speaker/car audio, and is not a production solution.

## 5. Engineering verification

1. Run `python -m unittest tests.test_alerts` and `python -m unittest tests.test_multimodal`
2. Run `python -m multimodal.alerts --test-speech`; verify sound from intended device while stationary
3. Send exactly one test SMS using `--send-test-sms`; verify arrival and actual read-aloud behaviour on the recipient phone while parked
4. Disconnect internet and verify that sending fails visibly but camera processing continues
5. Trigger a controlled warning in an injected/mock test; verify once-only and cooldown behaviour
6. On real hardware, verify bad signals do not cause warnings and timestamp delays are recorded

## Safety and privacy

- Never send SMS to a phone number without the recipient's consent
- Do not automatically contact emergency services or other people from unvalidated model output
- Avoid putting names, locations, participant identifiers or medical details in messages
- Use consent and suitable access control for any stored human EEG or camera records
