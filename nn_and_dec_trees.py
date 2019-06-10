    def sentence_to_vector(self, sentence, features):
        sentence_features = {}
        for feature in features:
            sentence_features.update(feature(sentence))
        sorted_keys = sorted(features)
        arr = np.ndarray(len(features))
        for i, key in enumerate(sorted_keys):
            arr[i] = sentence_features[key]
        return arr