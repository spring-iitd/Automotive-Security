class BaseSplitter:
    def __init__(self, input_dir, feature_extractor="PixNet", **kwargs):
        self.input_dir = input_dir
        self.feature_extractor = feature_extractor
        self.kwargs = kwargs

    def split(self):
        raise NotImplementedError("Subclasses must implement this method.")
