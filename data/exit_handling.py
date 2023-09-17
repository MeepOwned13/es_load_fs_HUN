import sys
import atexit


class ExitHooks(object):
    def __init__(self):
        self.exit_code = 0
        self.exception = None
        self._orig_exit = sys.exit
        self._orig_exc_handler = self.exc_handler

    def hook(self):
        sys.exit = self.exit
        sys.excepthook = self.exc_handler

    def exit(self, code=0):
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, *args):
        self.exception = exc
        self._orig_exc_handler(self, exc_type, exc, *args)


exit_hooks = ExitHooks()
exit_hooks.hook()
