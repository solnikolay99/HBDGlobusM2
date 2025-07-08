class BaseModel:
    def __init__(self):
        self._mObservers = []

    def add_observer(self, inObserver):
        self._mObservers.append(inObserver)

    def remove_observer(self, inObserver):
        self._mObservers.remove(inObserver)

    def notify_observers(self):
        for x in self._mObservers:
            x.model_is_changed()
