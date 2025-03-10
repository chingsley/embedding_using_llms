using System;

public class Person
{
    // Private fields
    private string name;
    private int age;

    // Constructor
    public Person(string name, int age)
    {
        this.name = name;
        this.age = age;
    }

    // Getter for name
    public string GetName()
    {
        return name;
    }

    // Setter for name
    public void SetName(string name)
    {
        this.name = name;
    }

    // Getter for age
    public int GetAge()
    {
        return age;
    }

    // Setter for age
    public void SetAge(int age)
    {
        this.age = age;
    }

    // Method to display person details
    public void DisplayDetails()
    {
        Console.WriteLine($"Name: {name}, Age: {age}");
    }

    // Main method for testing
    public static void Main(string[] args)
    {
        Person person = new Person("John Doe", 30);
        person.DisplayDetails();
    }
}