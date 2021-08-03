# To facilitate running the same code on different machine or cluster.
# This should abstract this by taking a config as input and simply running on the machines specified by the config.
# By default the config should loach locally
# It should support submitit if installed
# It should be implementable as a subclass or could directly launch a function.


class Launcher:
    def __init__():
        raise NotImplementedError

    def launch():
        raise NotImplementedError
