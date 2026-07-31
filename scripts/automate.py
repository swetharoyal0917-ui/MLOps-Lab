import os

print("=" * 50)
print("Reusable Automation Scripts")
print("=" * 50)

print("\nRunning Tests...")
os.system("python -m pytest")

print("\nBuilding Docker Image...")
os.system("docker build -t reusable_automation .")

print("\nDeploying Application...")
os.system("docker run --rm reusable_automation")

print("\nAutomation Finished Successfully.")