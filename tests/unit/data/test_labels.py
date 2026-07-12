from quanxin_life.data.labels import DiagnosticPoint, derive_eol80


def test_eol80_requires_three_persistent_low_points() -> None:
    points = [
        DiagnosticPoint(cycle_index=100, soh=0.79),
        DiagnosticPoint(cycle_index=110, soh=0.81),
        DiagnosticPoint(cycle_index=120, soh=0.79),
        DiagnosticPoint(cycle_index=130, soh=0.78),
        DiagnosticPoint(cycle_index=140, soh=0.77),
    ]

    label = derive_eol80(points)

    assert label.eol_cycle == 120
    assert label.confirmation_cycles == (120, 130, 140)
    assert label.right_censored is False


def test_eol80_is_right_censored_without_confirmation() -> None:
    label = derive_eol80(
        [DiagnosticPoint(cycle_index=100, soh=0.79), DiagnosticPoint(cycle_index=110, soh=0.78)]
    )

    assert label.eol_cycle is None
    assert label.right_censored is True


def test_invalid_diagnostics_do_not_confirm_eol() -> None:
    label = derive_eol80(
        [
            DiagnosticPoint(cycle_index=100, soh=0.79),
            DiagnosticPoint(cycle_index=110, soh=0.78, valid=False),
            DiagnosticPoint(cycle_index=120, soh=0.77),
            DiagnosticPoint(cycle_index=130, soh=0.76),
        ]
    )

    assert label.confirmation_cycles == (100, 120, 130)

