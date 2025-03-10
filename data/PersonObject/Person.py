class Person:
    # Constructor
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Getter for name
    def get_name(self):
        return self.name

    # Setter for name
    def set_name(self, name):
        self.name = name

    # Getter for age
    def get_age(self):
        return self.age

    # Setter for age
    def set_age(self, age):
        self.age = age

    # Method to display person details
    def display_details(self):
        print(f"Name: {self.name}, Age: {self.age}")

# Testing the class
if __name__ == "__main__":
    person = Person("John Doe", 30)
    person.display_details()