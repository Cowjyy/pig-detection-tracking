"""Frozen provenance accepts newline differences but never changed annotation values."""
import hashlib
import unittest
from scripts.restore_frozen_audit import frozen_bytes

class FrozenAuditPortabilityTests(unittest.TestCase):
    def test_cross_platform_serialization_recovers_original_bytes(self):
        windows=b'{\r\n  "width": 1920\r\n}\r\n'
        linux=windows.replace(b'\r\n',b'\n')
        expected=hashlib.sha256(windows).hexdigest()
        self.assertEqual(frozen_bytes(linux,expected),windows)
        self.assertEqual(frozen_bytes(windows,expected),windows)

    def test_changed_annotation_is_not_relabelled_as_the_original(self):
        original=b'{\n  "width": 1920\n}\n'
        changed=original.replace(b'1920',b'1280')
        with self.assertRaises(ValueError):
            frozen_bytes(changed,hashlib.sha256(original).hexdigest())
