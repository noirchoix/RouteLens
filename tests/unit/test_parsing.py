import numpy as np

from reacts.chemistry.conditions import reparse_multistep_middle, validate_conditions
from reacts.chemistry.reactions import parse_reaction
from reacts.contracts import ParseFailureClass
from reacts.data.parsing import parse_list, patent_document_id, stable_group_split


def test_symbolic_intermediate_is_not_conflated_with_invalid_smiles():
    parsed = parse_reaction("CCO>>M1")
    assert not parsed.parse_ok
    assert parsed.reactants_valid and parsed.products_valid
    assert parsed.failure_class == ParseFailureClass.SYMBOLIC_INTERMEDIATE


def test_malformed_reaction_and_valid_reaction():
    assert parse_reaction("CCO.CC(=O)O>>CCOC(C)=O").parse_ok
    assert parse_reaction("CCO>O").failure_class == ParseFailureClass.MALFORMED_DELIMITER


def test_list_decoding_and_patent_grouping():
    assert parse_list("['O', 'CO']") == ["O", "CO"]
    assert patent_document_id("20121206-US20120309739A1-0397") == "20121206-US20120309739A1"
    assert stable_group_split("20121206-US20120309739A1") == stable_group_split("20121206-US20120309739A1")


def test_condition_outliers_are_preserved_but_not_training_safe():
    result = validate_conditions(-2100, 61_700_000)
    assert result.temperature_observed_c == -2100
    assert result.temperature_clean_c is None
    assert result.time_observed_h == 61_700_000
    assert result.time_clean_h is None
    assert result.status == "suspicious"


def test_multistep_middle_repairs_temperature_time_order():
    parsed = reparse_multistep_middle("CCO>O.100.64800>CC=O")
    assert parsed.temperature_c == 100
    assert parsed.time_h == 18
    assert parsed.confidence == "high"



def test_list_decoding_preserves_numpy_parquet_arrays():
    assert parse_list(np.array(["O", "CO"], dtype=object)) == ["O", "CO"]
    assert parse_list(np.array([], dtype=object)) == []
