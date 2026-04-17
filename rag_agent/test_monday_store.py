import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_agent import monday_store


class MondayStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.orig_file = monday_store.MONDAY_CREDENTIALS_FILE
        monday_store.MONDAY_CREDENTIALS_FILE = Path(self.tmp.name) / "monday_credentials.json"

    def tearDown(self):
        monday_store.MONDAY_CREDENTIALS_FILE = self.orig_file
        self.tmp.cleanup()

    def test_oauth_state_roundtrip(self):
        state = monday_store.create_oauth_state("alice")
        self.assertTrue(state)
        username = monday_store.consume_oauth_state(state)
        self.assertEqual(username, "alice")
        self.assertIsNone(monday_store.consume_oauth_state(state))

    def test_credentials_isolated_per_user(self):
        monday_store.save_user_credentials("alice", {"access_token": "token_a"})
        monday_store.save_user_credentials("bob", {"access_token": "token_b"})
        alice = monday_store.get_user_credentials("alice")
        bob = monday_store.get_user_credentials("bob")
        self.assertEqual((alice or {}).get("access_token"), "token_a")
        self.assertEqual((bob or {}).get("access_token"), "token_b")


if __name__ == "__main__":
    unittest.main()
