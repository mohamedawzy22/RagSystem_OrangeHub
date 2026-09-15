import os
import random
import string

from helpers.config import get_setting


class BaseController:
    def __init__(self):
        self.app_setting = get_setting()
        self.base_dir = os.path.dirname(os.path.dirname(__file__))

        self.files_dir = os.path.join(self.base_dir, "asset/files")

    def generate_random_string(self, length: int = 12):
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
