class PreprocessorEditorRegistry:
    def __init__(self):
        self.registered = []

    def register(self, editor, priority=1000):
        self.registered.append((editor, priority))

    def sorted(self, category=None):
        if category is None:
            for editor, _ in sorted(self.registered, key=lambda x: x[1]):
                yield editor
        else:
            for editor, _ in sorted(self.registered, key=lambda x: x[1]):
                if category in editor.categories:
                    yield editor

    def categories(self):
        categories = set()
        for editor, _ in self.registered:
            categories.update(editor.categories)
        return sorted(categories)


preprocess_editors = PreprocessorEditorRegistry()
