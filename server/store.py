class DataStore:
    def __init__(self):
        self.data = {}

    def add_data(self, key, value):
        self.data[key] = value

    def get_data(self, key):
        return self.data.get(key)
    def get_keys(self):
        return self.data.keys()
    
