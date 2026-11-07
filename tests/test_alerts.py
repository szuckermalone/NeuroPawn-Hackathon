import unittest
from multimodal.alerts import AlertConfig, AlertDispatcher, validate_e164


class TestAlerts(unittest.TestCase):
    def test_phone_number_validation(self):
        self.assertEqual(validate_e164('+15551234567'), '+15551234567')
        for x in ('5551234567', '1234', '+1 abc', '+00000000000'):
            with self.assertRaises(ValueError): validate_e164(x)

    def test_edge_and_cooldown(self):
        t=[0.0]; sent=[]
        def sms(msg, phone): sent.append(('sms',phone)); return 'fake-test-id'
        def say(msg): sent.append(('speech',msg))
        a=AlertDispatcher(AlertConfig(True,True,'+15551234567',300),sms_sender=sms,speaker=say,clock=lambda:t[0])
        try:
            self.assertFalse(a.update('MONITORING',genuine_hardware=True,validated_inputs=True))
            self.assertFalse(a.update('EXPERIMENTAL WARNING',genuine_hardware=False,validated_inputs=True))
            self.assertFalse(a.update('EXPERIMENTAL WARNING',genuine_hardware=True,validated_inputs=False))
            a.update('MONITORING',genuine_hardware=True,validated_inputs=True)
            self.assertTrue(a.update('EXPERIMENTAL WARNING',genuine_hardware=True,validated_inputs=True))
            self.assertFalse(a.update('EXPERIMENTAL WARNING',genuine_hardware=True,validated_inputs=True))
            t[0]=60.0
            a.update('MONITORING',genuine_hardware=True,validated_inputs=True)
            self.assertFalse(a.update('EXPERIMENTAL WARNING',genuine_hardware=True,validated_inputs=True))
            t[0]=301.0
            a.update('MONITORING',genuine_hardware=True,validated_inputs=True)
            self.assertTrue(a.update('EXPERIMENTAL WARNING',genuine_hardware=True,validated_inputs=True))
        finally: a.close()
        self.assertEqual(len([x for x in sent if x[0]=='sms']),2)
        self.assertEqual(len([x for x in sent if x[0]=='speech']),2)

    def test_does_not_alert_without_opt_in(self):
        a=AlertDispatcher(AlertConfig(),sms_sender=lambda *x:self.fail('unexpected SMS'),speaker=lambda *x:self.fail('unexpected voice'))
        try:self.assertFalse(a.update('EXPERIMENTAL WARNING',genuine_hardware=True,validated_inputs=True))
        finally:a.close()

if __name__=='__main__':unittest.main()
