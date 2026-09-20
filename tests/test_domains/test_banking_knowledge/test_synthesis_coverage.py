from tau3.synthesis.catalog import build_catalog
from tau3.synthesis.validation import validate_static


def test_replacements_keep_combined_opening_closure_coverage():
    import tau3.synthesis.scenarios as module

    catalog = build_catalog()
    # The failed pilot replacement previously switched this slot to closure-only.
    replacement = module.sample_candidate(catalog, 11141 + 1000003, 11, 1)
    assert replacement.skeleton.private["open_business"] is True
    validate_static(replacement, catalog)
    for attempt in range(3):
        for count in (20, 200):
            items = [
                module.sample_candidate(
                    catalog, 42 + slot * 1009 + attempt * 1000003, slot, attempt
                )
                for slot in range(3, count, 4)
            ]
            assert (
                sum(r.skeleton.private["open_business"] for r in items) >= count // 20
            )
