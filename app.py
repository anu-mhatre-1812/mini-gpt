"""Mini-GPT - AI Coding Assistant by Anuj Mhatre
Demo version with built-in responses.
"""

import os
import gradio as gr

CODE_RESPONSES = {
    "hello": "Hello! I'm Mini-GPT, your AI coding assistant. How can I help?",
    "reverse string": "```python\ndef reverse_string(s):\n    return s[::-1]\n\nprint(reverse_string('hello'))  # 'olleh'\n```",
    "fibonacci": "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nprint(fibonacci(10))  # 55\n```",
    "hello world": "```python\nprint('Hello, World!')\n```",
    "sort": "```python\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n\nprint(bubble_sort([64, 34, 25, 12, 22, 11, 90]))\n```",
    "factorial": "```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(5))  # 120\n```",
    "palindrome": "```python\ndef is_palindrome(s):\n    return s == s[::-1]\n\nprint(is_palindrome('racecar'))  # True\nprint(is_palindrome('hello'))    # False\n```",
    "binary search": "```python\ndef binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n```",
    "merge sort": "```python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n```",
    "api": "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Item(BaseModel):\n    name: str\n    price: float\n\n@app.get('/')\ndef root():\n    return {'message': 'Hello World'}\n\n@app.post('/items/')\ndef create_item(item: Item):\n    return {'name': item.name, 'price': item.price}\n```",
    "list comprehension": "```python\n# Basic list comprehension\nsquares = [x**2 for x in range(10)]\n# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]\n\n# With condition\nevens = [x for x in range(20) if x % 2 == 0]\n# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]\n```",
    "decorator": "```python\ndef timer(func):\n    import time\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        end = time.time()\n        print(f'{func.__name__} took {end-start:.4f}s')\n        return result\n    return wrapper\n\n@timer\ndef slow_function():\n    import time\n    time.sleep(1)\n    return 'done'\n```",
}

def chat(message, history):
    msg = message.lower()
    
    for key, response in CODE_RESPONSES.items():
        if key in msg:
            return response
    
    return f"Hi! I'm **Mini-GPT**, your AI coding assistant by Anuj Mhatre.\n\nTry asking me about:\n- Reverse a string\n- Fibonacci sequence\n- Binary search\n- Merge sort\n- Factorial\n- Palindrome check\n- FastAPI template\n- List comprehensions\n- Decorators\n- Hello world"

demo = gr.ChatInterface(
    fn=chat,
    title="Mini-GPT",
    description="AI Coding Assistant by Anuj Mhatre | Demo mode - Ask me to write code!",
    examples=[
        "Write a hello world in Python",
        "How to reverse a string?",
        "Show me binary search",
        "FastAPI template",
        "Decorators in Python",
    ],

)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
