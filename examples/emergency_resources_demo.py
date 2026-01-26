#!/usr/bin/env python3
"""Demo of emergency resources module functionality."""

from core.reporting.data import EmergencyResources, DisasterType


def main():
    """Demonstrate emergency resources module."""
    print("=" * 80)
    print("Emergency Resources Module Demo")
    print("=" * 80)
    print()

    resources = EmergencyResources()

    # Example 1: Get national resources
    print("1. NATIONAL EMERGENCY RESOURCES")
    print("-" * 80)
    national = resources.get_national_resources()
    for contact in national[:3]:  # Show first 3
        print(f"  • {contact.name}")
        if contact.phone:
            print(f"    Phone: {contact.phone}")
        if contact.url:
            print(f"    URL: {contact.url}")
        print(f"    {contact.description}")
        print()

    # Example 2: Get state resources
    print("\n2. STATE EMERGENCY MANAGEMENT (Florida)")
    print("-" * 80)
    state = resources.get_state_resources("FL")
    print(f"  State: {state.state_name}")
    print(f"  Emergency Mgmt: {state.emergency_mgmt_url}")
    print(f"  Phone: {state.emergency_mgmt_phone}")
    print(f"  Governor's Office: {state.governor_office_url}")
    print()

    # Example 3: Get disaster-specific resources
    print("\n3. HURRICANE-SPECIFIC RESOURCES")
    print("-" * 80)
    hurricane = resources.get_disaster_specific_resources(DisasterType.HURRICANE)
    for contact in hurricane:
        print(f"  • {contact.name}")
        if contact.url:
            print(f"    {contact.url}")
        print(f"    {contact.description}")
        print()

    # Example 4: Get road closure info
    print("\n4. ROAD CLOSURE INFORMATION")
    print("-" * 80)
    road_url = resources.get_road_closure_url("FL")
    print(f"  Florida DOT: {road_url}")
    print()

    # Example 5: Generate complete resources section
    print("\n5. COMPLETE RESOURCES SECTION (Florida Hurricane)")
    print("-" * 80)
    section = resources.generate_resources_section(
        state_abbrev="FL",
        disaster_type=DisasterType.HURRICANE,
        county_name="Miami-Dade"
    )

    print(f"  National Resources: {len(section['national'])} contacts")
    print(f"  State Resources: {section['state'].state_name}")
    print(f"  Disaster-Specific: {len(section['disaster_specific'])} contacts")
    print(f"  Road Info: {section['road_info_url']}")
    print(f"\n  What To Do ({len(section['what_to_do'])} actions):")
    for i, action in enumerate(section['what_to_do'][:3], 1):
        print(f"    {i}. {action}")
    print(f"    ... ({len(section['what_to_do']) - 3} more)")
    print()

    # Example 6: Different disaster types
    print("\n6. ACTION ITEMS BY DISASTER TYPE")
    print("-" * 80)
    for disaster_type in DisasterType:
        section = resources.generate_resources_section("CA", disaster_type)
        actions = section['what_to_do']
        print(f"  {disaster_type.value.title()}: {len(actions)} actions")
        if actions:
            print(f"    First: {actions[0]}")
    print()

    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
