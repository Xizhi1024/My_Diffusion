"""Check TotalSegmentator v2 API."""
import totalsegmentator.python_api as api
import inspect

# List all public names
public = [n for n in dir(api) if not n.startswith('_')]
print("=== Public names in totalsegmentator.python_api ===")
for n in public:
    print(f"  {n}")

# Check for functions
print("\n=== Functions ===")
for n in public:
    obj = getattr(api, n)
    if callable(obj) and not inspect.isclass(obj):
        try:
            sig = inspect.signature(obj)
            print(f"  {n}{sig}")
        except:
            print(f"  {n}(...)")

print("\n=== Classes ===")
for n in public:
    obj = getattr(api, n)
    if inspect.isclass(obj):
        print(f"  {n}")
