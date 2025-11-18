#!/usr/bin/env python3
"""
Script to generate all remaining course notebooks
"""
import json
import os

def create_notebook(title, sections):
    """Create a Jupyter notebook structure"""
    cells = []
    
    # Title
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [f'# {title}\n\n---\n']
    })
    
    # Sections
    for section in sections:
        if 'markdown' in section:
            cells.append({
                'cell_type': 'markdown',
                'metadata': {},
                'source': section['markdown']
            })
        if 'code' in section:
            cells.append({
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': section['code']
            })
    
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.10.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }

# Define all notebooks
notebooks = {
    # INTERMEDIATE
    '02-intermediate/01-oop-basics.ipynb': {
        'title': '🐍 Intermediate: OOP Basics',
        'sections': [
            {'markdown': ['## 📦 Classes and Objects\n']},
            {'code': ['class Person:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n    \n    def greet(self):\n        return f"Hi, I\'m {self.name}"\n\np = Person("Alice", 25)\nprint(p.greet())']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Classes define objects\n✅ `__init__` is constructor\n✅ `self` refers to instance\n']}
        ]
    },
    '02-intermediate/02-file-handling.ipynb': {
        'title': '🐍 Intermediate: File Handling',
        'sections': [
            {'markdown': ['## 📄 Reading and Writing Files\n']},
            {'code': ['# Writing to a file\nwith open("test.txt", "w") as f:\n    f.write("Hello, World!")\n\n# Reading from a file\nwith open("test.txt", "r") as f:\n    content = f.read()\n    print(content)']},
            {'markdown': ['## 📊 CSV Files\n']},
            {'code': ['import csv\n\n# Writing CSV\nwith open("data.csv", "w", newline="") as f:\n    writer = csv.writer(f)\n    writer.writerow(["Name", "Age"])\n    writer.writerow(["Alice", 25])']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Use `with` for file handling\n✅ Modes: r, w, a, r+\n✅ CSV module for structured data\n']}
        ]
    },
    '02-intermediate/03-error-handling.ipynb': {
        'title': '🐍 Intermediate: Error Handling',
        'sections': [
            {'markdown': ['## ⚠️ Try-Except Blocks\n']},
            {'code': ['try:\n    x = 10 / 0\nexcept ZeroDivisionError:\n    print("Cannot divide by zero!")\nexcept Exception as e:\n    print(f"Error: {e}")\nfinally:\n    print("Cleanup code here")']},
            {'markdown': ['## 🛑 Raising Exceptions\n']},
            {'code': ['def check_age(age):\n    if age < 0:\n        raise ValueError("Age cannot be negative")\n    return age\n\ntry:\n    check_age(-5)\nexcept ValueError as e:\n    print(e)']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ try-except for error handling\n✅ `finally` always executes\n✅ `raise` to throw exceptions\n']}
        ]
    },
    '02-intermediate/04-modules-packages.ipynb': {
        'title': '🐍 Intermediate: Modules & Packages',
        'sections': [
            {'markdown': ['## 📦 Importing Modules\n']},
            {'code': ['import math\nfrom datetime import datetime\nimport random as rnd\n\nprint(math.pi)\nprint(datetime.now())\nprint(rnd.randint(1, 10))']},
            {'markdown': ['## 📚 Standard Library\n']},
            {'code': ['import os\nimport sys\nimport json\n\nprint(f"Python version: {sys.version}")\nprint(f"Current directory: {os.getcwd()}")']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ `import` to use modules\n✅ Standard library is powerful\n✅ Use `pip` to install packages\n']}
        ]
    },
    '02-intermediate/05-decorators-generators.ipynb': {
        'title': '🐍 Intermediate: Decorators & Generators',
        'sections': [
            {'markdown': ['## ⭐ Decorators\n']},
            {'code': ['def my_decorator(func):\n    def wrapper():\n        print("Before")\n        func()\n        print("After")\n    return wrapper\n\n@my_decorator\ndef say_hello():\n    print("Hello!")\n\nsay_hello()']},
            {'markdown': ['## 🔄 Generators\n']},
            {'code': ['def count_up_to(n):\n    count = 1\n    while count <= n:\n        yield count\n        count += 1\n\nfor num in count_up_to(5):\n    print(num)']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Decorators modify functions\n✅ `yield` creates generators\n✅ Generators are memory efficient\n']}
        ]
    },
    
    # ADVANCED
    '03-advanced/01-advanced-oop.ipynb': {
        'title': '🐍 Advanced: Advanced OOP',
        'sections': [
            {'markdown': ['## 🏗️ Inheritance\n']},
            {'code': ['class Animal:\n    def __init__(self, name):\n        self.name = name\n    def speak(self):\n        pass\n\nclass Dog(Animal):\n    def speak(self):\n        return f"{self.name} says Woof!"\n\nclass Cat(Animal):\n    def speak(self):\n        return f"{self.name} says Meow!"\n\ndog = Dog("Buddy")\nprint(dog.speak())']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Inheritance for code reuse\n✅ `super()` calls parent methods\n✅ Polymorphism allows flexibility\n']}
        ]
    },
    '03-advanced/02-multithreading-multiprocessing.ipynb': {
        'title': '🐍 Advanced: Concurrency',
        'sections': [
            {'markdown': ['## 🔀 Threading\n']},
            {'code': ['import threading\nimport time\n\ndef worker(name):\n    print(f"{name} starting")\n    time.sleep(2)\n    print(f"{name} done")\n\nthreads = []\nfor i in range(3):\n    t = threading.Thread(target=worker, args=(f"Worker-{i}",))\n    threads.append(t)\n    t.start()\n\nfor t in threads:\n    t.join()']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Threading for I/O-bound tasks\n✅ Multiprocessing for CPU-bound\n✅ async/await for modern concurrency\n']}
        ]
    },
    '03-advanced/03-context-managers.ipynb': {
        'title': '🐍 Advanced: Context Managers',
        'sections': [
            {'markdown': ['## 📝 Custom Context Managers\n']},
            {'code': ['class FileManager:\n    def __init__(self, filename):\n        self.filename = filename\n    \n    def __enter__(self):\n        self.file = open(self.filename, "w")\n        return self.file\n    \n    def __exit__(self, exc_type, exc_val, exc_tb):\n        self.file.close()\n\nwith FileManager("test.txt") as f:\n    f.write("Hello!")']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ `__enter__` and `__exit__` methods\n✅ Ensures cleanup\n✅ `with` statement automatic\n']}
        ]
    },
    '03-advanced/04-metaclasses.ipynb': {
        'title': '🐍 Advanced: Metaclasses',
        'sections': [
            {'markdown': ['## 🔮 Metaclasses\n']},
            {'code': ['class Meta(type):\n    def __new__(cls, name, bases, dct):\n        print(f"Creating class {name}")\n        return super().__new__(cls, name, bases, dct)\n\nclass MyClass(metaclass=Meta):\n    pass']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Metaclasses create classes\n✅ Advanced Python feature\n✅ Use sparingly\n']}
        ]
    },
    '03-advanced/05-performance-optimization.ipynb': {
        'title': '🐍 Advanced: Performance',
        'sections': [
            {'markdown': ['## ⚡ Profiling Code\n']},
            {'code': ['import time\n\ndef slow_function():\n    time.sleep(0.1)\n    return sum(range(1000000))\n\nstart = time.time()\nresult = slow_function()\nend = time.time()\nprint(f"Time: {end - start:.4f}s")']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Profile before optimizing\n✅ Use appropriate data structures\n✅ List comprehensions are fast\n']}
        ]
    },
}

# Create all notebooks
for filepath, config in notebooks.items():
    nb = create_notebook(config['title'], config['sections'])
    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'✅ Created {filepath}')

print('\n🎉 Generated all Intermediate and Advanced notebooks!')
print('📝 Note: ML, DL, and GenAI notebooks require more detailed content.')
print('    Use COURSE-OUTLINE.md as reference for those sections.')
