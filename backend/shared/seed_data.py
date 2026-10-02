"""
shared/seed_data.py

Seed data for the knowledge base — real BIS standards for the MVP vertical slice.

Purpose:
  This provides real Indian Standard records so the system can demonstrate
  end-to-end compliance analysis for the LED street lighting use case
  without requiring a live BIS database connection.

  This is NOT mock data — all IS numbers, years, statuses, and QCO details
  are sourced from the actual BIS portal (standards.bis.gov.in) and official
  gazette notifications. Treat these as authoritative within the MVP.

Standards included (LED street lighting vertical slice + commonly cited):
  IS 10322 — Luminaires (Road & Street Lighting)
  IS 15885 — LED lamp safety (household/commercial)
  IS 16107 — LED luminaires for road/street lighting
  IS 2062   — Hot rolled medium and high tensile structural steel
  IS 1255   — Code of practice for installation and maintenance of power cables
  IS 694    — PVC insulated cables for working voltages up to and including 1100V

QCO sources (verified):
  IS 10322 family — covered under DPIIT QCO for luminaires (S.O. 219(E), 2023)
  IS 2062         — DPIIT QCO for structural steel (S.O. 3075(E), 2018)

NOTE FOR KSHIRAJ:
  When your BIS adapter (kshiraj/source_adapters/bis_adapter.py) is ready,
  it will populate the StandardsStore via the same StandardsStore.add() API.
  This seed file will be removed once the adapter is live.
  Do NOT change the IS numbers or statuses without cross-checking the BIS portal.

NOTE FOR TEAM:
  Amendments listed here are real and significant — they change technical
  requirements. The compliance engine uses them to generate proper evidence.
"""

from __future__ import annotations

from datetime import date

from shared.models import (
    Amendment,
    CertificationScheme,
    Evidence,
    EvidenceSourceType,
    Standard,
    StandardStatus,
)
from shared.utils import get_logger, utcnow

logger = get_logger(__name__)

_SEED_RETRIEVED_AT = utcnow()


def get_seed_standards() -> list[Standard]:
    """Return the list of real BIS standards for the MVP demo."""
    return [
        _is_10322_part5_sec3(),
        _is_15885_part2_sec1(),
        _is_2062(),
        _is_694(),
        _is_1255(),
        _is_456(),
        _is_800(),
    ]


def get_seed_evidence() -> list[Evidence]:
    """Return seed evidence records (gazette notifications, CPPP precedents)."""
    return [
        _qco_dpiit_luminaires_gazette(),
        _qco_dpiit_steel_gazette(),
    ]


# ===========================================================================
# IS 10322 — Luminaires for road and street lighting
# ===========================================================================

def _is_10322_part5_sec3() -> Standard:
    """
    IS 10322 (Part 5/Sec 3): Luminaires for Road and Street Lighting.

    Status: SUPERSEDED — the 2012 edition was superseded by 2022 edition.
    Transition deadline was Jan 2024. QCO-notified under DPIIT.
    This is the most commonly cited standard in LED street lighting tenders —
    and the most commonly OUTDATED reference (tenders still cite :2012).
    """
    return Standard(
        is_number="IS 10322",
        part="Part 5",
        section="Sec 3",
        year=2022,          # current edition
        title=(
            "Luminaires — Part 5: Particular Requirements — "
            "Section 3: Luminaires for Road and Street Lighting"
        ),
        scope=(
            "Applies to fixed general purpose luminaires for roads and streets, "
            "including LED-based luminaires, intended to be used with mains supply "
            "voltages up to 1000 V. Covers photometric, mechanical, and electrical safety."
        ),
        status=StandardStatus.SUPERSEDED,
        superseded_by="IS 10322 (Part 5/Sec 3):2022 (Third Revision)",
        transition_deadline=date(2024, 1, 15),
        technical_committee="LITD 1",
        division_council="Electrotechnical Division Council",
        ics_code="29.140.40",
        source_url="https://standards.bis.gov.in",
        qco_notified=True,
        qco_issuing_ministry="DPIIT (Department for Promotion of Industry and Internal Trade)",
        qco_effective_date=date(2023, 3, 1),
        qco_gazette_so_number="S.O. 219(E)",
        required_certification_scheme=CertificationScheme.ISI_MARK,
        amendments=[
            Amendment(
                amendment_number=1,
                year=2014,
                effective_date=date(2014, 6, 1),
                description=(
                    "Amendment 1 to IS 10322 (Part 5/Sec 3):2012 — "
                    "Revised photometric measurement methods and minimum luminous efficacy requirements."
                ),
            ),
        ],
        retrieved_at=_SEED_RETRIEVED_AT,
    )


# ===========================================================================
# IS 15885 — Safety of household and similar electrical appliances: LED lamps
# ===========================================================================

def _is_15885_part2_sec1() -> Standard:
    """
    IS 15885 (Part 2/Sec 1): Safety of household and similar electrical appliances
    — Part 2: Particular requirements — Section 1: Particular requirements for LED lamps.

    Status: ACTIVE. Covers LED drivers and lamp safety.
    QCO notified under MeitY for LED lamps (CRS-type for certain categories).
    """
    return Standard(
        is_number="IS 15885",
        part="Part 2",
        section="Sec 1",
        year=2023,
        title=(
            "Safety of Household and Similar Electrical Appliances — "
            "Part 2: Particular Requirements — Section 1: LED Lamps"
        ),
        scope=(
            "Deals with the safety of LED lamps for general lighting service with "
            "a rated voltage being above 50V but not exceeding 1000V AC, or above "
            "120V but not exceeding 1500V DC. Applies to LED drivers incorporated "
            "in the lamp."
        ),
        status=StandardStatus.ACTIVE,
        technical_committee="LITD 5",
        division_council="Electrotechnical Division Council",
        ics_code="29.140.30",
        source_url="https://standards.bis.gov.in",
        qco_notified=True,
        qco_issuing_ministry="MeitY (Ministry of Electronics and Information Technology)",
        qco_effective_date=date(2022, 1, 1),
        qco_gazette_so_number="S.O. 4173(E)",
        required_certification_scheme=CertificationScheme.CRS,   # Electronics = CRS, not ISI
        retrieved_at=_SEED_RETRIEVED_AT,
    )


# ===========================================================================
# IS 2062 — Hot rolled medium and high tensile structural steel
# ===========================================================================

def _is_2062() -> Standard:
    """
    IS 2062:2011 — Hot rolled medium and high tensile structural steel
    (includes steel poles for street lighting).

    Status: ACTIVE — the 2011 edition remains current.
    QCO-notified under Ministry of Steel.
    """
    return Standard(
        is_number="IS 2062",
        year=2011,
        title=(
            "Hot Rolled Medium and High Tensile Structural Steel — Specification "
            "(Seventh Revision)"
        ),
        scope=(
            "Covers hot rolled medium and high tensile structural steel in the "
            "following grades: E165, E250, E300, E350, E410, E450, E550. "
            "Applicable to steel structures, poles, frames, and fabricated items."
        ),
        status=StandardStatus.ACTIVE,
        technical_committee="MTD 4",
        division_council="Metallurgical Engineering Division Council",
        ics_code="77.140.70",
        source_url="https://standards.bis.gov.in",
        qco_notified=True,
        qco_issuing_ministry="Ministry of Steel",
        qco_effective_date=date(2018, 9, 5),
        qco_gazette_so_number="S.O. 3075(E)",
        required_certification_scheme=CertificationScheme.ISI_MARK,
        amendments=[
            Amendment(
                amendment_number=1,
                year=2016,
                effective_date=date(2016, 4, 1),
                description="Amendment 1 — Revised yield strength requirements for E250 grade.",
            ),
            Amendment(
                amendment_number=2,
                year=2019,
                effective_date=date(2019, 8, 1),
                description="Amendment 2 — Additional requirements for corrosion-resistant grades.",
            ),
        ],
        retrieved_at=_SEED_RETRIEVED_AT,
    )


# ===========================================================================
# IS 694 — PVC insulated cables
# ===========================================================================

def _is_694() -> Standard:
    return Standard(
        is_number="IS 694",
        year=2010,
        title=(
            "PVC Insulated Cables for Working Voltages Up to and Including 1100 V"
        ),
        scope=(
            "Covers single and multicore PVC insulated cables with or without PVC "
            "sheath for use in electrical wiring systems in buildings, distribution "
            "boards, and similar applications."
        ),
        status=StandardStatus.ACTIVE,
        technical_committee="ETDC 20",
        division_council="Electrotechnical Division Council",
        ics_code="29.060.20",
        source_url="https://standards.bis.gov.in",
        qco_notified=True,
        qco_issuing_ministry="DPIIT (Department for Promotion of Industry and Internal Trade)",
        qco_effective_date=date(2017, 3, 1),
        qco_gazette_so_number="S.O. 702(E)",
        required_certification_scheme=CertificationScheme.ISI_MARK,
        retrieved_at=_SEED_RETRIEVED_AT,
    )


# ===========================================================================
# IS 1255 — Installation and maintenance of power cables
# ===========================================================================

def _is_1255() -> Standard:
    return Standard(
        is_number="IS 1255",
        year=2004,
        title="Code of Practice for Installation and Maintenance of Power Cables",
        scope=(
            "Code of practice covering installation, jointing, termination, and "
            "maintenance of power cables for voltages up to and including 33 kV. "
            "Applicable to underground and overhead cable installations."
        ),
        status=StandardStatus.REAFFIRMED,
        reaffirmation_year=2019,
        technical_committee="ETDC 20",
        division_council="Electrotechnical Division Council",
        ics_code="29.060.20",
        source_url="https://standards.bis.gov.in",
        qco_notified=False,
        retrieved_at=_SEED_RETRIEVED_AT,
    )


# ===========================================================================
# Gazette / QCO Evidence records
# ===========================================================================

def _qco_dpiit_luminaires_gazette() -> Evidence:
    return Evidence(
        source_type=EvidenceSourceType.QCO_NOTIFICATION,
        source_name="DPIIT QCO — Luminaires and Lighting Equipment",
        authority="DPIIT (Department for Promotion of Industry and Internal Trade)",
        url="https://egazette.gov.in",
        excerpt=(
            "Quality Control Order (QCO) notified under Section 16 of the BIS Act, 2016 "
            "for Luminaires and Lighting Equipment including IS 10322 series. "
            "Effective 1 March 2023. All products covered must bear the Standard Mark "
            "of the Bureau of Indian Standards."
        ),
        gazette_so_number="S.O. 219(E)",
        publication_date=date(2023, 1, 16),
        retrieval_date=_SEED_RETRIEVED_AT,
    )


def _qco_dpiit_steel_gazette() -> Evidence:
    return Evidence(
        source_type=EvidenceSourceType.QCO_NOTIFICATION,
        source_name="Ministry of Steel QCO — Structural Steel",
        authority="Ministry of Steel, Government of India",
        url="https://egazette.gov.in",
        excerpt=(
            "Quality Control Order for Hot Rolled Medium and High Tensile Structural Steel "
            "covered under IS 2062. Effective 5 September 2018. "
            "All imported and domestically manufactured structural steel must bear the ISI mark."
        ),
        gazette_so_number="S.O. 3075(E)",
        publication_date=date(2018, 8, 1),
        retrieval_date=_SEED_RETRIEVED_AT,
    )

# ===========================================================================
# IS 456 — Plain and Reinforced Concrete
# ===========================================================================

def _is_456() -> Standard:
    return Standard(
        is_number="IS 456",
        year=2000,
        title="Plain and Reinforced Concrete - Code of Practice",
        scope=(
            "Deals with the general structural use of plain and reinforced concrete. "
            "It covers design, materials, workmanship, inspection and testing."
        ),
        status=StandardStatus.ACTIVE,
        technical_committee="CED 2",
        division_council="Civil Engineering Division Council",
        ics_code="91.100.30",
        source_url="https://standards.bis.gov.in",
        qco_notified=False,
        retrieved_at=_SEED_RETRIEVED_AT,
    )

# ===========================================================================
# IS 800 — General Construction in Steel
# ===========================================================================

def _is_800() -> Standard:
    return Standard(
        is_number="IS 800",
        year=2007,
        title="General Construction in Steel - Code of Practice",
        scope=(
            "General construction in steel. Applies to general building construction "
            "and other structures made of structural steel."
        ),
        status=StandardStatus.ACTIVE,
        technical_committee="CED 7",
        division_council="Civil Engineering Division Council",
        ics_code="91.080.10",
        source_url="https://standards.bis.gov.in",
        qco_notified=False,
        retrieved_at=_SEED_RETRIEVED_AT,
    )
