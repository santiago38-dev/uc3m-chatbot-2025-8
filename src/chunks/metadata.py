"""
Metadata Registry - Load Enriched CSV with Intelligence Layer

Loads the Gold Master CSV (enriched_145_projects_SURGICAL.csv) and applies:
1. Hardcoded Parent Company Mapping (100% capture for top developers)
2. Zone Mapping (spatial intelligence from county)

Author: Santiago (UC3M Applied AI)
Date: December 2025
"""

import csv
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field
import re


# =============================================================================
# INTELLIGENCE LAYER: HARDCODED PARENT MAPPING
# Guarantees 100% capture for top developers that control 80%+ of MWs
# =============================================================================

PARENT_MAPPING = {
    # Tier 1: The Giants (MUST capture 100%)
    # Include full SPV names with LLC/Inc suffixes as they appear in ERCOT docs
    'NEXTERA': ['nextera', 'fpl', 'florida power', 'logan p', 'nextera energy', 'nextera energy resources'],
    'RWE': ['rwe', 'e.on', 'inland', 'rwe clean', 'rwe solar', 'rwe renewables',
            'rwe solar development', 'rwe renewables americas'],
    'INVENERGY': ['invenergy', 'invenergy llc', 'invenergy solar'],
    'EDF': ['edf', 'edf renewables', 'edf renewable'],
    'ENEL': ['enel', 'enel green', 'enel green power', 'enel north america'],
    'AES': ['aes ', 'aes clean energy', 'aes renewable'],
    'ENGIE': ['engie', 'engie north america', 'engie storage'],
    'ORSTED': ['orsted', 'ørsted', 'orsted onshore'],
    'AVANGRID': ['avangrid', 'avangrid renewables'],
    'PATTERN': ['pattern energy', 'pattern '],
    'EDP': ['edp renewables', 'edp ', 'edp renewables north america'],
    'TERRA-GEN': ['terra-gen', 'terra gen', 'terragen'],

    # Tier 2: Major Players
    'CANADIAN SOLAR': ['canadian solar', 'recurrent', 'recurrent energy'],
    'LIGHTSOURCE BP': ['lightsource', 'bp ', 'lightsource bp'],
    'CIP': ['copenhagen', 'copenhagen infrastructure', 'cip'],
    'SAVION': ['savion', 'savion llc'],
    'OCI': ['oci ', 'alamo', 'oci solar'],
    'CLEARWAY': ['clearway', 'clearway energy'],
    'APEX': ['apex clean', 'apex ', 'apex clean energy'],
    'HECATE': ['hecate', 'hecate energy', 'hecate grid'],
    'LEEWARD': ['leeward', 'leeward renewable'],
    'LONGROAD': ['longroad', 'longroad energy'],
    'CYPRESS CREEK': ['cypress creek', 'cypress creek renewables'],
    '174 POWER': ['174 power', '174 power global'],
    'LINCOLN': ['lincoln clean energy', 'lincoln clean'],
    'ADAPTURE': ['adapture', 'adapture renewables'],
    'IP ENERGY': ['ip quantum', 'ip energy'],
    'ENBRIDGE': ['enbridge', 'enbridge inc'],

    # Tier 3: Battery/Storage Specialists
    'PLUS POWER': ['plus power', 'plus power llc'],
    'KEY CAPTURE': ['key capture', 'key capture energy'],
    'BROAD REACH': ['broad reach', 'broad reach power'],
    'JUPITER': ['jupiter power', 'jupiter '],
    'ABLE GRID': ['able grid', 'able grid energy'],
    'GIGA TEXAS': ['giga texas'],
    'SAMSUNG': ['samsung', 'samsung c&t', 'samsung sdi', 'samsung engineering', 'samsung e&a'],
    'HANWHA': ['hanwha', 'hanwha energy', 'hanwha q cells', 'hanwha solutions'],
    'SK': ['sk e&s', 'sk energy', 'sk ecoplant'],
    'FLUENCE': ['fluence', 'fluence energy'],
    'WARTSILA': ['wartsila', 'wärtsilä'],
    'POWIN': ['powin', 'powin energy'],

    # Tier 4: Utilities & IPPs
    'VISTRA': ['vistra'],
    'NRG': ['nrg '],
    'DUKE': ['duke '],
    'SOUTHERN': ['southern company', 'southern power'],
    'DOMINION': ['dominion'],
    'LCRA': ['lcra'],
    'CPS': ['cps energy'],

    # Tier 5: PE/Infrastructure
    'CPV': ['cpv', 'competitive power'],
    'BLACKROCK': ['blackrock'],
    'BROOKFIELD': ['brookfield'],
    'KKR': ['kkr'],
    'CARLYLE': ['carlyle'],

    # Tier 6: Regional/Other Known
    'GRANSOLAR': ['gransolar'],
    'MISAE': ['misae', 'excel advantage'],
    'VESPER': ['vesper'],
    'OPEN ROAD': ['open road'],
    'ORIGIS': ['origis'],
    'SOL SYSTEMS': ['sol systems'],
    'TRI GLOBAL': ['tri global', 'tri-global'],
    'SILICON RANCH': ['silicon ranch'],
    '8MINUTE': ['8minute', '8 minute'],
    'INTERSECT': ['intersect power'],
    'TAI': ['tai norton', 'tai '],
    'LUPINUS': ['lupinus solar'],
    'TRES BAHIAS': ['tres bahias'],
    'STARR SOLAR': ['starr solar'],
}


# =============================================================================
# INTELLIGENCE LAYER: ZONE MAPPING
# Maps ERCOT counties to load zones for regional cost analysis
# =============================================================================

COUNTY_ZONES = {
    # WEST TEXAS (Low congestion, high wind/solar)
    'PECOS': 'WEST', 'REEVES': 'WEST', 'WARD': 'WEST', 'ECTOR': 'WEST',
    'MIDLAND': 'WEST', 'HOWARD': 'WEST', 'MARTIN': 'WEST', 'ANDREWS': 'WEST',
    'CRANE': 'WEST', 'UPTON': 'WEST', 'WINKLER': 'WEST', 'LOVING': 'WEST',
    'CULBERSON': 'WEST', 'JEFF DAVIS': 'WEST', 'PRESIDIO': 'WEST',
    'BREWSTER': 'WEST', 'TERRELL': 'WEST', 'VAL VERDE': 'WEST',
    'CROCKETT': 'WEST', 'SUTTON': 'WEST', 'SCHLEICHER': 'WEST',
    'IRION': 'WEST', 'REAGAN': 'WEST', 'GLASSCOCK': 'WEST',
    'STERLING': 'WEST', 'COKE': 'WEST', 'TOM GREEN': 'WEST',

    # PANHANDLE (High wind, transmission constrained)
    'POTTER': 'PANHANDLE', 'RANDALL': 'PANHANDLE', 'CARSON': 'PANHANDLE',
    'DEAF SMITH': 'PANHANDLE', 'OLDHAM': 'PANHANDLE', 'HARTLEY': 'PANHANDLE',
    'MOORE': 'PANHANDLE', 'HUTCHINSON': 'PANHANDLE', 'SHERMAN': 'PANHANDLE',
    'HANSFORD': 'PANHANDLE', 'OCHILTREE': 'PANHANDLE', 'LIPSCOMB': 'PANHANDLE',
    'HEMPHILL': 'PANHANDLE', 'ROBERTS': 'PANHANDLE', 'GRAY': 'PANHANDLE',
    'WHEELER': 'PANHANDLE', 'ARMSTRONG': 'PANHANDLE', 'DONLEY': 'PANHANDLE',
    'COLLINGSWORTH': 'PANHANDLE', 'BRISCOE': 'PANHANDLE', 'HALL': 'PANHANDLE',
    'CHILDRESS': 'PANHANDLE', 'SWISHER': 'PANHANDLE', 'CASTRO': 'PANHANDLE',
    'PARMER': 'PANHANDLE', 'BAILEY': 'PANHANDLE', 'LAMB': 'PANHANDLE',
    'HALE': 'PANHANDLE', 'FLOYD': 'PANHANDLE', 'MOTLEY': 'PANHANDLE',
    'COTTLE': 'PANHANDLE', 'KING': 'PANHANDLE', 'DICKENS': 'PANHANDLE',

    # COAST (Houston load center, premium pricing)
    'HARRIS': 'COAST', 'BRAZORIA': 'COAST', 'GALVESTON': 'COAST',
    'FORT BEND': 'COAST', 'WHARTON': 'COAST', 'MATAGORDA': 'COAST',
    'JACKSON': 'COAST', 'VICTORIA': 'COAST', 'CALHOUN': 'COAST',
    'REFUGIO': 'COAST', 'ARANSAS': 'COAST', 'SAN PATRICIO': 'COAST',
    'NUECES': 'COAST', 'KLEBERG': 'COAST', 'KENEDY': 'COAST',
    'WILLACY': 'COAST', 'CAMERON': 'COAST', 'HIDALGO': 'COAST',
    'STARR': 'COAST',

    # NORTH (Dallas load center)
    'DALLAS': 'NORTH', 'TARRANT': 'NORTH', 'COLLIN': 'NORTH',
    'DENTON': 'NORTH', 'ELLIS': 'NORTH', 'JOHNSON': 'NORTH',
    'KAUFMAN': 'NORTH', 'ROCKWALL': 'NORTH', 'HUNT': 'NORTH',
    'COOKE': 'NORTH', 'GRAYSON': 'NORTH', 'FANNIN': 'NORTH',
    'LAMAR': 'NORTH', 'RED RIVER': 'NORTH', 'BOWIE': 'NORTH',
    'WISE': 'NORTH', 'JACK': 'NORTH', 'PARKER': 'NORTH',
    'HOOD': 'NORTH', 'SOMERVELL': 'NORTH', 'ERATH': 'NORTH',
    'PALO PINTO': 'NORTH', 'STEPHENS': 'NORTH', 'YOUNG': 'NORTH',
    'CLAY': 'NORTH', 'MONTAGUE': 'NORTH', 'WICHITA': 'NORTH',
    'WILBARGER': 'NORTH', 'HARDEMAN': 'NORTH', 'FOARD': 'NORTH',

    # SOUTH (San Antonio / Valley)
    'BEXAR': 'SOUTH', 'COMAL': 'SOUTH', 'GUADALUPE': 'SOUTH',
    'WILSON': 'SOUTH', 'ATASCOSA': 'SOUTH', 'MEDINA': 'SOUTH',
    'UVALDE': 'SOUTH', 'ZAVALA': 'SOUTH', 'FRIO': 'SOUTH',
    'LA SALLE': 'SOUTH', 'MCMULLEN': 'SOUTH', 'LIVE OAK': 'SOUTH',
    'BEE': 'SOUTH', 'GOLIAD': 'SOUTH', 'KARNES': 'SOUTH',
    'DE WITT': 'SOUTH', 'GONZALES': 'SOUTH', 'CALDWELL': 'SOUTH',
    'HAYS': 'SOUTH', 'TRAVIS': 'SOUTH', 'WILLIAMSON': 'SOUTH',
    'WEBB': 'SOUTH', 'DIMMIT': 'SOUTH', 'MAVERICK': 'SOUTH',
    'KINNEY': 'SOUTH', 'EDWARDS': 'SOUTH', 'REAL': 'SOUTH',
    'BANDERA': 'SOUTH', 'KERR': 'SOUTH', 'KENDALL': 'SOUTH',

    # CENTRAL (Austin/Waco corridor)
    'MCLENNAN': 'CENTRAL', 'BELL': 'CENTRAL', 'CORYELL': 'CENTRAL',
    'LAMPASAS': 'CENTRAL', 'BURNET': 'CENTRAL', 'LLANO': 'CENTRAL',
    'MASON': 'CENTRAL', 'GILLESPIE': 'CENTRAL', 'BLANCO': 'CENTRAL',
    'BASTROP': 'CENTRAL', 'LEE': 'CENTRAL', 'FAYETTE': 'CENTRAL',
    'COLORADO': 'CENTRAL', 'AUSTIN': 'CENTRAL', 'WASHINGTON': 'CENTRAL',
    'BRAZOS': 'CENTRAL', 'BURLESON': 'CENTRAL', 'MILAM': 'CENTRAL',
    'FALLS': 'CENTRAL', 'LIMESTONE': 'CENTRAL', 'FREESTONE': 'CENTRAL',
    'LEON': 'CENTRAL', 'ROBERTSON': 'CENTRAL', 'MADISON': 'CENTRAL',
    'GRIMES': 'CENTRAL', 'WALKER': 'CENTRAL', 'MONTGOMERY': 'CENTRAL',
    'HILL': 'CENTRAL', 'NAVARRO': 'CENTRAL', 'HENDERSON': 'CENTRAL',
    'ANDERSON': 'CENTRAL', 'CHEROKEE': 'CENTRAL', 'NACOGDOCHES': 'CENTRAL',
    'FRANKLIN': 'CENTRAL', 'TITUS': 'CENTRAL', 'MORRIS': 'CENTRAL',
    'HASKELL': 'CENTRAL', 'KNOX': 'CENTRAL', 'BAYLOR': 'CENTRAL',
    'THROCKMORTON': 'CENTRAL', 'SHACKELFORD': 'CENTRAL', 'JONES': 'CENTRAL',
    'TAYLOR': 'CENTRAL', 'CALLAHAN': 'CENTRAL', 'EASTLAND': 'CENTRAL',
    'COMANCHE': 'CENTRAL', 'HAMILTON': 'CENTRAL', 'BOSQUE': 'CENTRAL',
    'NOLAN': 'CENTRAL', 'MITCHELL': 'CENTRAL', 'SCURRY': 'CENTRAL',
    'FISHER': 'CENTRAL', 'STONEWALL': 'CENTRAL', 'KENT': 'CENTRAL',
    'GARZA': 'CENTRAL', 'LYNN': 'CENTRAL', 'DAWSON': 'CENTRAL',
    'BORDEN': 'CENTRAL', 'RUNNELS': 'CENTRAL', 'COLEMAN': 'CENTRAL',
    'BROWN': 'CENTRAL', 'MILLS': 'CENTRAL', 'SAN SABA': 'CENTRAL',
    'MCCULLOCH': 'CENTRAL', 'MENARD': 'CENTRAL', 'CONCHO': 'CENTRAL',
}


# =============================================================================
# PROJECT METADATA DATACLASS
# =============================================================================

@dataclass
class ProjectMetadata:
    """Complete project metadata from enriched CSV + intelligence layer."""

    # Identifiers
    inr: str = ""
    item_number: str = ""

    # From CSV (pre-loaded)
    project_name: str = ""
    developer_spv: str = ""        # Raw SPV name from CSV
    parent_company: str = ""       # INTELLIGENCE: Normalized parent
    fuel_type: str = ""            # WIN, SOL, OTH, GAS
    technology: str = ""           # WT, PV, BESS, etc.
    capacity_mw: float = 0.0
    county: str = ""
    zone: str = ""                 # INTELLIGENCE: Mapped zone
    cdr_zone: str = ""             # From CSV (CDR Reporting Zone)

    # Dates
    projected_cod: str = ""
    ia_signed: str = ""
    fis_approved: str = ""
    file_date: str = ""

    # Document info
    doc_type: str = ""
    has_final_sgia: bool = False
    description: str = ""

    # Status
    maturity_score: int = 0
    is_amended: bool = False       # Detected from doc_type/description

    # To be extracted from PDF
    tsp_name: str = ""
    security_total_usd: float = 0.0
    security_design_usd: float = 0.0
    security_construction_usd: float = 0.0
    security_per_kw: float = 0.0
    commercial_operation_date: str = ""

    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            'inr': self.inr,
            'item_number': self.item_number,
            'project_name': self.project_name,
            'developer_spv': self.developer_spv,
            'parent_company': self.parent_company,
            'fuel_type': self.fuel_type,
            'technology': self.technology,
            'capacity_mw': self.capacity_mw,
            'county': self.county,
            'zone': self.zone,
            'cdr_zone': self.cdr_zone,
            'projected_cod': self.projected_cod,
            'ia_signed': self.ia_signed,
            'fis_approved': self.fis_approved,
            'file_date': self.file_date,
            'doc_type': self.doc_type,
            'has_final_sgia': self.has_final_sgia,
            'description': self.description,
            'maturity_score': self.maturity_score,
            'is_amended': self.is_amended,
            'tsp_name': self.tsp_name,
            'security_total_usd': self.security_total_usd,
            'security_design_usd': self.security_design_usd,
            'security_construction_usd': self.security_construction_usd,
            'security_per_kw': self.security_per_kw,
            'commercial_operation_date': self.commercial_operation_date,
        }


# =============================================================================
# METADATA REGISTRY
# =============================================================================

class MetadataRegistry:
    """
    Load and manage project metadata from enriched CSV.

    Applies intelligence layer:
    - Parent company normalization (100% capture for top developers)
    - Zone mapping (spatial intelligence)
    """

    def __init__(self):
        self.registry: Dict[str, ProjectMetadata] = {}
        self.by_item: Dict[str, ProjectMetadata] = {}

    def _infer_parent_company(self, spv_name: str) -> str:
        """
        Apply hardcoded parent mapping to SPV name.

        Args:
            spv_name: Raw interconnecting entity name (with trailing spaces stripped)

        Returns:
            Normalized parent company name or 'UNKNOWN'
        """
        if not spv_name:
            return 'UNKNOWN'

        spv_lower = spv_name.lower()

        for parent, keywords in PARENT_MAPPING.items():
            if any(kw in spv_lower for kw in keywords):
                return parent

        return 'UNKNOWN'

    def _get_zone(self, county: str, cdr_zone: str = "") -> str:
        """
        Map county to ERCOT zone.

        Uses CDR Reporting Zone from CSV if available, otherwise maps from county.

        Args:
            county: County name
            cdr_zone: CDR Reporting Zone from CSV (if available)

        Returns:
            Zone name (WEST, PANHANDLE, COAST, NORTH, SOUTH, CENTRAL, OTHER)
        """
        # If CDR zone is provided and valid, use it
        if cdr_zone and cdr_zone.upper() in ['WEST', 'PANHANDLE', 'COAST', 'NORTH', 'SOUTH', 'CENTRAL']:
            return cdr_zone.upper()

        # Otherwise map from county
        if not county:
            return 'OTHER'

        county_upper = county.upper().strip()
        return COUNTY_ZONES.get(county_upper, 'OTHER')

    def _detect_amendment(self, doc_type: str, description: str) -> bool:
        """Detect if document is an amendment from doc_type or description."""
        text = f"{doc_type} {description}".lower()
        return 'amended' in text or 'restated' in text or 'amendment' in text

    def load_enriched_csv(self, csv_path: Path) -> int:
        """
        Load the Gold Master enriched CSV and apply intelligence layer.

        Args:
            csv_path: Path to enriched_145_projects_SURGICAL.csv

        Returns:
            Number of projects loaded
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"Enriched CSV not found: {csv_path}")

        count = 0

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Get raw values (strip trailing spaces!)
                inr = row.get('INR', '').strip()
                item = row.get('Item', '').strip()
                spv_raw = row.get('Interconnecting Entity', '').strip()
                county = row.get('County', '').strip()
                cdr_zone = row.get('CDR Reporting Zone', '').strip()
                doc_type = row.get('Doc_Type', '').strip()
                description = row.get('Description', '').strip()

                # Apply intelligence layer
                parent_company = self._infer_parent_company(spv_raw)
                zone = self._get_zone(county, cdr_zone)
                is_amended = self._detect_amendment(doc_type, description)

                # Parse capacity
                try:
                    capacity = float(row.get('Capacity (MW)', 0) or 0)
                except (ValueError, TypeError):
                    capacity = 0.0

                # Parse maturity score
                try:
                    maturity = int(row.get('Maturity_Score', 0) or 0)
                except (ValueError, TypeError):
                    maturity = 0

                # Create metadata object
                meta = ProjectMetadata(
                    inr=inr,
                    item_number=item,
                    project_name=row.get('Project', '').strip(),
                    developer_spv=spv_raw,
                    parent_company=parent_company,
                    fuel_type=row.get('Fuel', '').strip(),
                    technology=row.get('Technology', '').strip(),
                    capacity_mw=capacity,
                    county=county,
                    zone=zone,
                    cdr_zone=cdr_zone,
                    projected_cod=row.get('Projected COD', '').strip(),
                    ia_signed=row.get('IA Signed', '').strip(),
                    fis_approved=row.get('FIS Approved', '').strip(),
                    file_date=row.get('File_Date', '').strip(),
                    doc_type=doc_type,
                    has_final_sgia=row.get('Has_Final_SGIA', '').strip().lower() == 'true',
                    description=description,
                    maturity_score=maturity,
                    is_amended=is_amended,
                )

                # Store by INR and Item
                if inr:
                    self.registry[inr] = meta
                if item:
                    self.by_item[item] = meta

                count += 1

        print(f"✓ Loaded {count} projects from {csv_path.name}")

        # Print intelligence summary
        parent_counts = {}
        zone_counts = {}
        for meta in self.registry.values():
            parent_counts[meta.parent_company] = parent_counts.get(meta.parent_company, 0) + 1
            zone_counts[meta.zone] = zone_counts.get(meta.zone, 0) + 1

        known_parents = sum(c for p, c in parent_counts.items() if p != 'UNKNOWN')
        print(f"  Parent Company: {known_parents}/{count} identified ({100*known_parents/count:.0f}%)")
        print(f"  Zones: {dict(sorted(zone_counts.items(), key=lambda x: -x[1]))}")

        return count

    def get_by_inr(self, inr: str) -> Optional[ProjectMetadata]:
        """Get metadata by INR number."""
        return self.registry.get(inr)

    def get_by_item(self, item: str) -> Optional[ProjectMetadata]:
        """Get metadata by Item number."""
        return self.by_item.get(item)

    def get_by_filename(self, filename: str) -> Optional[ProjectMetadata]:
        """
        Get metadata by PDF filename.

        Filename format: 35077_{Item}_{FileID}.pdf
        """
        match = re.search(r'35077_(\d+)_\d+', filename)
        if match:
            item = match.group(1)
            return self.by_item.get(item)
        return None

    def get_all(self) -> list:
        """Get all project metadata."""
        return list(self.registry.values())

    def __len__(self):
        return len(self.registry)
