"""Unit tests for pymstm._parser -- text-only, no MSTM binary required."""

from pymstm._parser import parse_mstm_output

# Real MSTM output snippets (captured verbatim from a two-sphere run),
# trimmed to just the lines _parse_total_efficiencies cares about.

_PLAIN_TOTAL_BLOCK = """\
 number iterations, error, solution time
     4  4.1994E-08  1.6926E-02
 total extinction, absorption, scattering efficiencies (unpol, par, perp incidence)
  1.3203E+00  3.1301E-01  1.0073E+00  1.3203E+00  3.1301E-01  1.0073E+00  1.3203E+00  3.1301E-01  1.0073E+00
 down and up hemispherical scattering efficiencies (unpol, par, perp)
  2.6944E-01  7.3799E-01  2.6944E-01  7.3799E-01  2.6944E-01  7.3799E-01
"""

_INCIDENCE_AVERAGED_TOTAL_BLOCK = """\
 number iterations, error, solution time
     4  6.5538E-08  1.3175E-01
 total extinction, absorption, scattering efficiencies (unpolarized incidence)
  4.7645E+00  3.3468E-01  4.4298E+00
  down and up hemispherical scattering efficiencies (unpol, par, perp)
  3.3985E-01  4.0816E+00  3.4282E-01  4.0869E+00  3.3688E-01  4.0762E+00
"""


def test_parse_total_efficiencies_plain(tmp_path):
    p = tmp_path / "plain.dat"
    p.write_text(_PLAIN_TOTAL_BLOCK)
    result = parse_mstm_output(p)
    assert result["total"] == {
        "q_ext_unpol": 1.3203,
        "q_abs_unpol": 0.31301,
        "q_sca_unpol": 1.0073,
        "q_ext_par": 1.3203,
        "q_abs_par": 0.31301,
        "q_sca_par": 1.0073,
        "q_ext_perp": 1.3203,
        "q_abs_perp": 0.31301,
        "q_sca_perp": 1.0073,
    }


def test_parse_total_efficiencies_incidence_averaged(tmp_path):
    """incidence_average=true reports only the unpolarized triple, on its
    own differently-worded header line -- regression test for the gap
    where _parse_total_efficiencies's 9-value-only regex silently
    returned {} for this format."""
    p = tmp_path / "incidence_averaged.dat"
    p.write_text(_INCIDENCE_AVERAGED_TOTAL_BLOCK)
    result = parse_mstm_output(p)
    assert result["total"] == {
        "q_ext_unpol": 4.7645,
        "q_abs_unpol": 0.33468,
        "q_sca_unpol": 4.4298,
    }
    assert "q_ext_par" not in result["total"]


def test_parse_total_efficiencies_missing_returns_empty(tmp_path):
    p = tmp_path / "empty.dat"
    p.write_text("nothing relevant here\n")
    result = parse_mstm_output(p)
    assert result["total"] == {}
