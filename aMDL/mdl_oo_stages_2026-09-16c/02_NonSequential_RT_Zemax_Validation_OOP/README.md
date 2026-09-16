# 02_NonSequential_RT_Zemax_Validation_OOP — NSC validation (nscval 2026-09-16.01)

The non-sequential companion of `02_Sequential_RT_Zemax_Validation_OOP`
(renamed from `02_validation_zemax_oo` on 2026-09-16; its `zval` package
supplies the OpticStudio connection and the log to this stage).

    02_NonSequential_RT_Zemax_Validation_OOP\
        mdl_nsc_validation.py    CLI: probe | null | ladder  [--gui] [options]
        NSC_TRACK.md             the plan: rungs 1-7, why NSC, facts on file
        nscval\
            settings.py          TRACE / NULL / LADDER / PROBE settings (all echoed)
            dlls.py              verbatim slot maps of srg_blaze / srg_step
            tea.py               scalar references (blaze sinc^2, staircase orders)
            nsc.py               NscSystem (NCE builder), DiffractionTab (name
                                 adapter, slots -> Reflect + Transmit), NscTrace
            base.py              RunContext, NscAnalysis (<run>\nsc\<stamp>_<mode>)
            probe.py             rung 1   nulltest.py  rung 2   ladder.py  rung 3
        tests\
            mock_nsc.py          ZOS-API NSC stand-in (scalar 'ray trace')
            test_mock.py         the three modes end to end on the mock -> ALL OK

Status: written against the ZOS-API member names of the 2021-2024 NCE
examples; the Diffraction-tab members are resolved at run time through
candidate lists and are the first thing `probe` verifies on the real
build. Mock regression: null PASS (scalar vs scalar), ladder ratio 1.000.
`mypy` / `pyflakes` clean.
