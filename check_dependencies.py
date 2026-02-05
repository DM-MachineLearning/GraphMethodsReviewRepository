#!/usr/bin/env python3
"""
Dependency and environment validation script.

Checks that all required packages are installed with compatible versions.

Usage:
    python check_dependencies.py

Output:
    Prints status of each package and overall compatibility
"""

import sys
import importlib
from typing import Dict, Tuple


# Required packages with minimum versions
REQUIRED_PACKAGES = {
    'numpy': '1.16.0',
    'scipy': '1.2.0',
    'scikit-learn': '0.20.0',
    'tensorflow': '1.13.0',  # TF 1.x or 2.x accepted
}

# Optional packages
OPTIONAL_PACKAGES = {
    'matplotlib': '3.0.0',
    'tensorboard': '1.13.0',
    'jupyter': '1.0.0',
}


def parse_version(version_string: str) -> Tuple[int, int, int]:
    """Parse version string to tuple of integers."""
    parts = version_string.split('.')[:3]
    # Pad with zeros if needed
    while len(parts) < 3:
        parts.append('0')
    
    try:
        return tuple(int(p.split('a')[0].split('b')[0].split('rc')[0]) for p in parts)
    except (ValueError, AttributeError):
        return (0, 0, 0)


def check_package(package_name: str, min_version: str, required: bool = True) -> Dict:
    """
    Check if a package is installed with sufficient version.
    
    Args:
        package_name: Name of package (import name)
        min_version: Minimum required version string
        required: Whether package is required (True) or optional (False)
    
    Returns:
        Dict with status information
    """
    try:
        # Handle package name variations
        import_name = package_name
        if package_name == 'scikit-learn':
            import_name = 'sklearn'
        
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        # Parse versions for comparison
        if version != 'unknown':
            installed = parse_version(version)
            required_v = parse_version(min_version)
            version_ok = installed >= required_v
        else:
            version_ok = True  # Can't check, assume OK
        
        return {
            'name': package_name,
            'installed': True,
            'version': version,
            'version_ok': version_ok,
            'min_required': min_version,
            'required': required,
        }
    
    except ImportError:
        return {
            'name': package_name,
            'installed': False,
            'version': None,
            'version_ok': False,
            'min_required': min_version,
            'required': required,
        }


def print_status(results: list) -> None:
    """Print dependency check results."""
    print("\n" + "="*80)
    print("DEPENDENCY CHECK")
    print("="*80 + "\n")
    
    # Separate required and optional
    required = [r for r in results if r['required']]
    optional = [r for r in results if not r['required']]
    
    # Print required packages
    print("REQUIRED PACKAGES:")
    print("-" * 80)
    all_required_ok = True
    
    for result in required:
        if result['installed']:
            status = "✓" if result['version_ok'] else "✗"
            print(f"{status} {result['name']:<20} {result['version']:<15} "
                  f"(required: {result['min_required']})")
            if not result['version_ok']:
                all_required_ok = False
        else:
            print(f"✗ {result['name']:<20} NOT INSTALLED")
            all_required_ok = False
    
    # Print optional packages
    if optional:
        print("\nOPTIONAL PACKAGES:")
        print("-" * 80)
        
        for result in optional:
            if result['installed']:
                status = "✓"
                print(f"{status} {result['name']:<20} {result['version']:<15} "
                      f"(required: {result['min_required']})")
            else:
                print(f"○ {result['name']:<20} NOT INSTALLED")
    
    # Summary
    print("\n" + "="*80)
    if all_required_ok:
        print("✓ ALL REQUIRED PACKAGES OK - Ready to use!")
    else:
        print("✗ MISSING REQUIRED PACKAGES - Install with:")
        print("  pip install -r requirements.txt")
    print("="*80 + "\n")
    
    return all_required_ok


def suggest_install() -> None:
    """Suggest installation commands for missing packages."""
    print("\nTo install missing packages, run:")
    print("  pip install -r requirements.txt")
    print("\nOr install individually:")
    print("  pip install numpy scipy scikit-learn tensorflow")
    print("\nFor GPU support, use:")
    print("  pip install tensorflow-gpu")


def main():
    """Main function."""
    results = []
    
    # Check required packages
    for package, min_version in REQUIRED_PACKAGES.items():
        results.append(check_package(package, min_version, required=True))
    
    # Check optional packages
    for package, min_version in OPTIONAL_PACKAGES.items():
        results.append(check_package(package, min_version, required=False))
    
    # Print results
    all_ok = print_status(results)
    
    # Return exit code
    return 0 if all_ok else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
