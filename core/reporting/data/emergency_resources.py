"""Emergency resource links and contact information for disaster response.

Provides national, state-specific, and disaster-type-specific emergency
resources including contact information, action items, and reference URLs.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class DisasterType(Enum):
    """Types of disasters supported by the platform."""

    FLOOD = "flood"
    WILDFIRE = "wildfire"
    HURRICANE = "hurricane"
    TORNADO = "tornado"
    EARTHQUAKE = "earthquake"


@dataclass
class EmergencyContact:
    """Emergency contact information."""

    name: str
    phone: Optional[str]
    url: Optional[str]
    description: str


@dataclass
class StateEmergencyInfo:
    """State-specific emergency management information."""

    state_name: str
    state_abbrev: str
    emergency_mgmt_url: str
    emergency_mgmt_phone: str
    governor_office_url: Optional[str] = None


# National-level "What To Do" action items by disaster type
FLOOD_ACTIONS = [
    "Do not walk or drive through flood waters",
    "If trapped, go to the highest level and call 911",
    "Document damage with photos for insurance claims",
    "Check on neighbors, especially elderly and disabled",
    "Boil water until authorities confirm safety",
    "Turn off utilities if instructed by authorities",
    "Stay away from downed power lines",
]

WILDFIRE_ACTIONS = [
    "Evacuate immediately if ordered by authorities",
    "Close all windows and doors if sheltering in place",
    "Wear N95 masks or respirators to protect from smoke",
    "Monitor air quality and stay indoors if unhealthy",
    "Keep emergency supplies and evacuation bag ready",
    "Document property for insurance claims",
    "Register with local emergency services if evacuated",
]

HURRICANE_ACTIONS = [
    "Evacuate if in an evacuation zone or ordered by authorities",
    "Board up windows and secure outdoor objects",
    "Stock emergency supplies for at least 7 days",
    "Fill bathtubs and containers with water",
    "Charge all electronic devices and batteries",
    "Stay indoors away from windows during the storm",
    "Do not go outside until authorities say it is safe",
]

TORNADO_ACTIONS = [
    "Seek shelter immediately in a basement or interior room",
    "Stay away from windows and cover yourself with blankets",
    "If in a mobile home, evacuate to a sturdy building",
    "If caught outside, lie flat in a ditch or low area",
    "Monitor weather radio or alerts continuously",
    "Have a tornado safety plan for every building you frequent",
    "After the storm, watch for hazards like downed power lines",
]

EARTHQUAKE_ACTIONS = [
    "Drop, Cover, and Hold On during shaking",
    "Stay away from windows, mirrors, and heavy objects",
    "If outside, move away from buildings and power lines",
    "After shaking stops, check for injuries and hazards",
    "Expect aftershocks and be ready to Drop, Cover, Hold On again",
    "Turn off gas if you smell gas or suspect a leak",
    "Do not use elevators; use stairs instead",
]


class EmergencyResources:
    """Provider of emergency resource links and contacts."""

    # National resources (always available)
    FEMA_PHONE = "1-800-621-3362"
    FEMA_URL = "https://www.fema.gov"
    FEMA_DISASTER_ASSISTANCE = "https://www.disasterassistance.gov"

    RED_CROSS_URL = "https://www.redcross.org"
    RED_CROSS_PHONE = "1-800-733-2767"
    RED_CROSS_SHELTER_FINDER = "https://www.redcross.org/get-help/disaster-relief-and-recovery-services/find-an-open-shelter.html"

    NWS_URL = "https://www.weather.gov"
    NWS_ALERTS = "https://www.weather.gov/alerts"

    NATIONAL_SUICIDE_PREVENTION = "988"  # Crisis support

    def __init__(self):
        """Initialize with state and disaster-specific data."""
        self._state_info = self._load_state_info()
        self._disaster_resources = self._load_disaster_resources()
        self._state_dot_urls = self._load_state_dot_urls()
        self._disaster_actions = self._load_disaster_actions()

    def get_national_resources(self) -> list[EmergencyContact]:
        """Get national emergency resources (FEMA, Red Cross, etc.)."""
        return [
            EmergencyContact(
                name="FEMA",
                phone=self.FEMA_PHONE,
                url=self.FEMA_URL,
                description="Federal Emergency Management Agency - disaster assistance and recovery"
            ),
            EmergencyContact(
                name="FEMA Disaster Assistance",
                phone=self.FEMA_PHONE,
                url=self.FEMA_DISASTER_ASSISTANCE,
                description="Apply for federal disaster assistance"
            ),
            EmergencyContact(
                name="American Red Cross",
                phone=self.RED_CROSS_PHONE,
                url=self.RED_CROSS_URL,
                description="Emergency shelter, supplies, and disaster relief"
            ),
            EmergencyContact(
                name="Red Cross Shelter Finder",
                phone=None,
                url=self.RED_CROSS_SHELTER_FINDER,
                description="Find open emergency shelters near you"
            ),
            EmergencyContact(
                name="National Weather Service",
                phone=None,
                url=self.NWS_URL,
                description="Official weather forecasts, warnings, and alerts"
            ),
            EmergencyContact(
                name="NWS Active Alerts",
                phone=None,
                url=self.NWS_ALERTS,
                description="Current weather alerts and warnings by location"
            ),
            EmergencyContact(
                name="988 Crisis Lifeline",
                phone=self.NATIONAL_SUICIDE_PREVENTION,
                url="https://988lifeline.org",
                description="Mental health crisis support and counseling"
            ),
        ]

    def get_state_resources(self, state_abbrev: str) -> StateEmergencyInfo:
        """Get state-specific emergency management info.

        Args:
            state_abbrev: Two-letter state code (e.g., 'FL', 'CA')

        Returns:
            StateEmergencyInfo for the state

        Raises:
            KeyError: If state not in database
        """
        state_upper = state_abbrev.upper()
        if state_upper not in self._state_info:
            raise KeyError(f"State {state_abbrev} not in emergency resources database")
        return self._state_info[state_upper]

    def get_disaster_specific_resources(
        self,
        disaster_type: DisasterType
    ) -> list[EmergencyContact]:
        """Get resources specific to disaster type.

        Args:
            disaster_type: Type of disaster

        Returns:
            List of disaster-specific emergency contacts
        """
        return self._disaster_resources.get(disaster_type, [])

    def get_road_closure_url(self, state_abbrev: str) -> str:
        """Get state DOT road closure information URL.

        Args:
            state_abbrev: Two-letter state code

        Returns:
            URL for state DOT road conditions/closures
        """
        state_upper = state_abbrev.upper()
        return self._state_dot_urls.get(
            state_upper,
            "https://www.fhwa.dot.gov/trafficinfo/"  # Federal fallback
        )

    def generate_resources_section(
        self,
        state_abbrev: str,
        disaster_type: DisasterType,
        county_name: Optional[str] = None
    ) -> dict:
        """Generate complete emergency resources section for a report.

        Args:
            state_abbrev: Two-letter state code
            disaster_type: Type of disaster
            county_name: Optional county name for additional context

        Returns:
            Dictionary with:
                - 'national': list[EmergencyContact]
                - 'state': StateEmergencyInfo
                - 'disaster_specific': list[EmergencyContact]
                - 'road_info_url': str
                - 'what_to_do': list[str]
        """
        return {
            'national': self.get_national_resources(),
            'state': self.get_state_resources(state_abbrev),
            'disaster_specific': self.get_disaster_specific_resources(disaster_type),
            'road_info_url': self.get_road_closure_url(state_abbrev),
            'what_to_do': self._disaster_actions.get(disaster_type, []),
        }

    def _load_state_info(self) -> dict[str, StateEmergencyInfo]:
        """Load state emergency management information.

        Returns:
            Dictionary mapping state abbreviation to StateEmergencyInfo
        """
        return {
            'FL': StateEmergencyInfo(
                state_name="Florida",
                state_abbrev="FL",
                emergency_mgmt_url="https://www.floridadisaster.org",
                emergency_mgmt_phone="1-800-342-3557",
                governor_office_url="https://www.flgov.com"
            ),
            'TX': StateEmergencyInfo(
                state_name="Texas",
                state_abbrev="TX",
                emergency_mgmt_url="https://tdem.texas.gov",
                emergency_mgmt_phone="512-424-2138",
                governor_office_url="https://gov.texas.gov"
            ),
            'CA': StateEmergencyInfo(
                state_name="California",
                state_abbrev="CA",
                emergency_mgmt_url="https://www.caloes.ca.gov",
                emergency_mgmt_phone="916-845-8510",
                governor_office_url="https://www.gov.ca.gov"
            ),
            'LA': StateEmergencyInfo(
                state_name="Louisiana",
                state_abbrev="LA",
                emergency_mgmt_url="https://gohsep.la.gov",
                emergency_mgmt_phone="225-925-7500",
                governor_office_url="https://gov.louisiana.gov"
            ),
            'NC': StateEmergencyInfo(
                state_name="North Carolina",
                state_abbrev="NC",
                emergency_mgmt_url="https://www.ncdps.gov/emergency-management",
                emergency_mgmt_phone="919-825-2500",
                governor_office_url="https://governor.nc.gov"
            ),
            'SC': StateEmergencyInfo(
                state_name="South Carolina",
                state_abbrev="SC",
                emergency_mgmt_url="https://www.scemd.org",
                emergency_mgmt_phone="803-737-8500",
                governor_office_url="https://governor.sc.gov"
            ),
            'GA': StateEmergencyInfo(
                state_name="Georgia",
                state_abbrev="GA",
                emergency_mgmt_url="https://gema.georgia.gov",
                emergency_mgmt_phone="404-635-7000",
                governor_office_url="https://gov.georgia.gov"
            ),
            'AL': StateEmergencyInfo(
                state_name="Alabama",
                state_abbrev="AL",
                emergency_mgmt_url="https://ema.alabama.gov",
                emergency_mgmt_phone="334-956-9200",
                governor_office_url="https://governor.alabama.gov"
            ),
            'MS': StateEmergencyInfo(
                state_name="Mississippi",
                state_abbrev="MS",
                emergency_mgmt_url="https://www.msema.org",
                emergency_mgmt_phone="601-933-6362",
                governor_office_url="https://www.ms.gov/governor"
            ),
            'VA': StateEmergencyInfo(
                state_name="Virginia",
                state_abbrev="VA",
                emergency_mgmt_url="https://www.vaemergency.gov",
                emergency_mgmt_phone="804-897-6500",
                governor_office_url="https://www.governor.virginia.gov"
            ),
            'NY': StateEmergencyInfo(
                state_name="New York",
                state_abbrev="NY",
                emergency_mgmt_url="https://www.dhses.ny.gov",
                emergency_mgmt_phone="518-242-5000",
                governor_office_url="https://www.governor.ny.gov"
            ),
            'NJ': StateEmergencyInfo(
                state_name="New Jersey",
                state_abbrev="NJ",
                emergency_mgmt_url="https://www.njoem.gov",
                emergency_mgmt_phone="609-882-2000",
                governor_office_url="https://nj.gov/governor"
            ),
        }

    def _load_disaster_resources(self) -> dict[DisasterType, list[EmergencyContact]]:
        """Load disaster-type-specific resources.

        Returns:
            Dictionary mapping DisasterType to list of EmergencyContact
        """
        return {
            DisasterType.FLOOD: [
                EmergencyContact(
                    name="National Flood Insurance Program",
                    phone="1-800-621-3362",
                    url="https://www.floodsmart.gov",
                    description="Flood insurance information and claims"
                ),
                EmergencyContact(
                    name="NOAA Flood Safety",
                    phone=None,
                    url="https://www.weather.gov/safety/flood",
                    description="Flood safety information and preparedness"
                ),
            ],
            DisasterType.WILDFIRE: [
                EmergencyContact(
                    name="National Interagency Fire Center",
                    phone="208-387-5512",
                    url="https://www.nifc.gov",
                    description="Wildfire status and incident information"
                ),
                EmergencyContact(
                    name="AirNow - Air Quality",
                    phone=None,
                    url="https://www.airnow.gov",
                    description="Real-time air quality data and forecasts"
                ),
                EmergencyContact(
                    name="InciWeb - Incident Information",
                    phone=None,
                    url="https://inciweb.nwcg.gov",
                    description="Wildfire incident details and updates"
                ),
            ],
            DisasterType.HURRICANE: [
                EmergencyContact(
                    name="National Hurricane Center",
                    phone=None,
                    url="https://www.nhc.noaa.gov",
                    description="Hurricane forecasts, warnings, and tracking"
                ),
                EmergencyContact(
                    name="FEMA Hurricane Preparedness",
                    phone=self.FEMA_PHONE,
                    url="https://www.ready.gov/hurricanes",
                    description="Hurricane preparation and safety guidance"
                ),
            ],
            DisasterType.TORNADO: [
                EmergencyContact(
                    name="Storm Prediction Center",
                    phone=None,
                    url="https://www.spc.noaa.gov",
                    description="Severe weather outlooks and warnings"
                ),
                EmergencyContact(
                    name="FEMA Tornado Safety",
                    phone=self.FEMA_PHONE,
                    url="https://www.ready.gov/tornadoes",
                    description="Tornado preparation and safety guidance"
                ),
            ],
            DisasterType.EARTHQUAKE: [
                EmergencyContact(
                    name="USGS Earthquake Hazards",
                    phone=None,
                    url="https://earthquake.usgs.gov",
                    description="Earthquake monitoring, maps, and data"
                ),
                EmergencyContact(
                    name="ShakeAlert",
                    phone=None,
                    url="https://www.shakealert.org",
                    description="Earthquake early warning system (West Coast)"
                ),
                EmergencyContact(
                    name="FEMA Earthquake Safety",
                    phone=self.FEMA_PHONE,
                    url="https://www.ready.gov/earthquakes",
                    description="Earthquake preparation and safety guidance"
                ),
            ],
        }

    def _load_state_dot_urls(self) -> dict[str, str]:
        """Load state Department of Transportation road info URLs.

        Returns:
            Dictionary mapping state abbreviation to DOT URL
        """
        return {
            'FL': "https://fl511.com",
            'TX': "https://drivetexas.org",
            'CA': "https://roads.dot.ca.gov",
            'LA': "https://511la.org",
            'NC': "https://drivenc.gov",
            'SC': "https://511sc.org",
            'GA': "https://511ga.org",
            'AL': "https://algotraffic.com",
            'MS': "https://mdottraffic.com",
            'VA': "https://511virginia.org",
            'NY': "https://511ny.org",
            'NJ': "https://511nj.org",
        }

    def _load_disaster_actions(self) -> dict[DisasterType, list[str]]:
        """Load disaster-specific action items.

        Returns:
            Dictionary mapping DisasterType to list of action items
        """
        return {
            DisasterType.FLOOD: FLOOD_ACTIONS,
            DisasterType.WILDFIRE: WILDFIRE_ACTIONS,
            DisasterType.HURRICANE: HURRICANE_ACTIONS,
            DisasterType.TORNADO: TORNADO_ACTIONS,
            DisasterType.EARTHQUAKE: EARTHQUAKE_ACTIONS,
        }
