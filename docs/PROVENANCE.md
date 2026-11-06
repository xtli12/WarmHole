# Provenance of the released files

The experiments ran from copies of these files on our server, many of them under versioned names. The left column is the file in this repository. The second column is the file that ran, as `$BH/...` on the server, with its md5. The Evidence column says how the md5 is known: *printed* = printed on the server; *uploaded* = identical to the copy we uploaded to the server; *local* = an analysis script run on downloaded results, not on the server (`make_specs.py` wrote the files in `specs/`, which were uploaded).

Changes of four kinds:

- **root**: the hard-coded project root became `$BH` (shell) or `os.environ['BH']` (Python), with the default `/data/blackhole`;
- **names**: references to versioned file names point to the single released file, e.g. `aw_train_c11.py` -> `aw_train.py`, `bh_client_lens17.py` -> `bh_client_lens.py`, and the process patterns `bh_client_lens1[2-7][.]py` -> `bh_client_lens[.]py`;
- **paths in specs**: `library` paths inside `specs/*.jsonl` now point to `/data/blackhole/lib`;
- **other**: listed in the row.

Code logic is unchanged. Where a released file consolidates several development versions, the **Supersedes** column lists them. Each earlier version is the released file with the newer options left off.

| File in this repository | Ran on the server as | md5 | Evidence | Changes | Supersedes |
|---|---|---|---|---|---|
| `code/bh_server.py` | `$BH/code/bh_server.py` | `0e115f282648` | uploaded | none |  |
| `code/bh_client.py` | `$BH/code/bh_client.py` | `30293f41db46` | printed | none |  |
| `code/wormhole.py` | `$BH/code/wormhole.py` | `7baa7339233c` | printed | none |  |
| `code/build_library.py` | `$BH/code/build_library.py` | `70ec72f175cd` | uploaded | none |  |
| `code/build_library2.py` | `$BH/code/build_library2.py` | `f06592b79ea9` | printed | none |  |
| `code/train_bridge.py` | `$BH/code/train_bridge.py` | `1ae2df7b4c80` | uploaded | none |  |
| `code/embed_instr.py` | `$BH/code/embed_instr.py` | `618bbfb1658a` | uploaded | none |  |
| `code/bridge_np.py` | `$BH/code/bridge_np.py` | `24073a116c26` | uploaded | none |  |
| `code/make_specs.py` | not run on the server | `4a381207d023` | local | root |  |
| `code/make_specs_w2.py` | `$BH/code/make_specs_w2.py` | `4d0e66cdd015` | printed | root |  |
| `code/bh_analyze.py` | `$BH/code/bh_analyze.py` | `0e77a4fc0067` | uploaded | none |  |
| `code/compute_anatomy.py` | not run on the server | `a1bc134eda0b` | local | none |  |
| `code/compute_tables.py` | `$BH/code/compute_tables.py` | `3188445bc79b` | uploaded | none |  |
| `code/relocation_stats.py` | not run on the server | `ec7709664a7e` | local | none |  |
| `code/summarize.py` | `$BH/code/summarize.py` | `efcaba89ade5` | uploaded | none |  |
| `code/w2_summary.py` | `$BH/code/w2_summary.py` | `3524447fa8ea` | printed | none |  |
| `code/status.py` | `$BH/code/status.py` | `d0e1da9c5881` | uploaded | none |  |
| `code/make_measured.py` | not run on the server | `d06f830e7be7` | local | none |  |
| `code/bench_overhead.py` | `$BH/code/bench_overhead.py` | `532550c7c6c6` | uploaded | none |  |
| `code/viz_runtime.py` | `$BH/code/viz_runtime.py` | `42cea22e237c` | uploaded | none |  |
| `code/patch_overlay.py` | `$BH/code/patch_overlay.py` | `db2c11e49b85` | uploaded | none |  |
| `code/dl_fast.py` | `$BH/code/dl_fast.py` | `d05650008470` | uploaded | none |  |
| `code/verify_ckpt.py` | `$BH/code/verify_ckpt.py` | `2cdc54048ac2` | uploaded | none |  |
| `code/w2_launch.sh` | `$BH/code/w2_launch.sh` | `36c230df1023` | printed | root |  |
| `code/w2_chain.sh` | `$BH/code/w2_chain.sh` | `efb8765c50ca` | printed | root |  |
| `code/w2_status.sh` | `$BH/code/w2_status.sh` | `19f807c149c6` | printed | root |  |
| `code/run_exp.sh` | `$BH/code/run_exp.sh` | `dd7ab2dd9ed2` | uploaded | root |  |
| `code/driver.sh` | `$BH/code/driver4.sh` | `90ad487aecf5` | uploaded | root; self-name driver4 -> driver |  |
| `code/setup_env.sh` | `$BH/code/setup_new.sh` | `a973aafe35e7` | uploaded | root; writes logs/client.ok after the LIBERO check (driver.sh waits for it; on the server it was created by hand) |  |
| `code/viz.sh` | `$BH/code/viz.sh` | `ac61e06741ff` | uploaded | root |  |
| `specs/pi05_e0.jsonl` | `$BH/specs/pi05_e0.jsonl` | `dd789c45d4ec` | uploaded | none |  |
| `specs/pi05_e1.jsonl` | `$BH/specs/pi05_e1.jsonl` | `03494ebfeb58` | uploaded | none |  |
| `specs/pi05_e2.jsonl` | `$BH/specs/pi05_e2.jsonl` | `8c75d26387a8` | uploaded | paths in specs |  |
| `specs/pi05_e3.jsonl` | `$BH/specs/pi05_e3.jsonl` | `72bf1e75f673` | uploaded | paths in specs |  |
| `specs/pi05_lib.jsonl` | `$BH/specs/pi05_lib.jsonl` | `8dc91e59fb5f` | uploaded | none |  |
| `specs/pi0_e0.jsonl` | `$BH/specs/pi0_e0.jsonl` | `095481255feb` | uploaded | none |  |
| `specs/pi0_e1.jsonl` | `$BH/specs/pi0_e1.jsonl` | `57318ca8fbf4` | uploaded | none |  |
| `specs/pi0_e2.jsonl` | `$BH/specs/pi0_e2.jsonl` | `0f50fbfe8adf` | uploaded | paths in specs |  |
| `specs/pi0_lib.jsonl` | `$BH/specs/pi0_lib.jsonl` | `d69c35def2a8` | uploaded | none |  |
| `specs/analysis_pairs.json` | `$BH/specs/analysis_pairs.json` | `5d86fb98d60e` | uploaded | none |  |
| `lens/lens_core.py` | `$BH/lens/lens_core.py` | `6830a9522130` | printed | none |  |
| `lens/lens_geom.py` | `$BH/lens/lens_geom.py` | `3f1b9f0af25d` | printed | none |  |
| `lens/lens_server.py` | `$BH/lens/lens_server.py` | `0af23a63b39c` | printed | none |  |
| `lens/lens_clean_errors.py` | `$BH/lens/lens_clean_errors.py` | `d8fc21474da7` | printed | none |  |
| `lens/patch_overlay_prefix.py` | `$BH/lens/patch_overlay_prefix.py` | `9a1a51f8f434` | printed | none |  |
| `lens/patch_overlay_heads.py` | `$BH/lens/patch_overlay_heads.py` | `e184a0787e0b` | printed | none |  |
| `lens/lens_launch.sh` | `$BH/lens/lens_launch.sh` | `64e02c30656c` | uploaded | root |  |
| `lens/lens_run.sh` | `$BH/lens/lens_run.sh` | `5276f712f397` | printed | root |  |
| `lens/lens_grasp_offset.py` | `$BH/lens/lens_grasp_offset.py` | `b8557b150a66` | printed | none |  |
| `lens/bh_client_lens.py` | `$BH/lens/bh_client_lens17.py` | `9760c7d5cf5b` | printed | none | lens11 (teacher data), lens12 (evaluation until 10-31), lens13 (DAgger), lens14 (evaluation from round e on), lens15 and lens16 (rounds g-i); every later version only adds behaviour that a spec key switches on |
| `lens/make_specs_lens.py` | `$BH/lens/make_specs_lens_r25.py` | `5ac42f179d7d` | printed | root; names | r13-r24; round j checked that r25 writes the same specs as r24 for every earlier call |
| `lens/aw/aw_gemma.py` | `$BH/lens/aw/aw_gemma.py` | `a92db349701a` | printed | none |  |
| `lens/aw/aw_model.py` | `$BH/lens/aw/aw_model_c10.py` | `8cb6cc0ca0a0` | printed | none | c1-c9 (deployed as `aw_model.py`: training and serving of every model); c10 adds the optional contrastive guidance and computes the same without it |
| `lens/aw/aw_train.py` | `$BH/lens/aw/aw_train_c11.py` | `8b04e51917aa` | printed | root | c1-c10 (deployed as `aw_train.py`; c10 differs from c9 only on pi0); c11 adds category-5 frames, per-frame weights and `--share`, identical sampling without them |
| `lens/aw/aw_build_data.py` | `$BH/lens/aw/aw_build_data_c7.py` | `911388c18fb6` | printed | none | c1-c6 (c4 deployed as `aw_build_data.py`); c5-c7 add `--self`, `--cat_as`, `--policy_max`, identical without them |
| `lens/aw/aw_server.py` | `$BH/lens/aw/aw_server_c9.py` | `4a0f97519db9` | printed | root; names | c1-c8 (deployed as `aw_server.py`); c9 adds the request mode `cig`, identical otherwise |
| `lens/aw/aw_eval.sh` | `$BH/lens/aw/aw_eval_c8.sh` | `8413e53afb29` | printed | root; names | c1-c7 (c6 deployed as `aw_eval.sh`); c7/c8 add the variables CLIENT, MS, AWSERVER and the splits selfplay, devr, devg, identical defaults otherwise |
| `lens/aw/aw_dagger.sh` | `$BH/lens/aw/aw_dagger.sh` | `0317c68e74b6` | printed | root; names |  |
| `lens/aw/dag_check.py` | `$BH/lens/aw/dag_check.py` | `aa5d85ab0752` | printed | none |  |
| `lens/aw/aw_summary.py` | `$BH/lens/aw/aw_summary.py` | `c9cdcb784d86` | printed | root |  |
| `lens/aw/aw_paired.py` | `$BH/lens/aw/aw_paired.py` | `a8dcf7ed6861` | printed | none |  |
| `lens/aw/aw_dev2.py` | `$BH/lens/aw/aw_dev2.py` | `714e7c0f118e` | printed | root |  |
| `lens/aw/aw_tables_local.py` | not run on the server | `d8558514dc8c` | local | none |  |
| `lens/aw/paper_rows.py` | not run on the server | `640acc2fbebb` | local | none |  |
| `lens/aw/lens_clean_errors.py` | `$BH/lens/aw/lens_clean_errors.py` | `d8fc21474da7` | printed | none |  |
| `lens/aw/qwen_heads.py` | `$BH/lens/aw/qwen_heads.py` | `467a51d87f95` | printed | root |  |
| `lens/aw/pi_heads.py` | `$BH/lens/aw/pi_heads.py` | `1d4e7060438a` | printed | root |  |
| `lens/aw/heads_small.py` | `$BH/lens/aw/heads_small.py` | `2cd695154ef7` | printed | none |  |
| `lens/aw/aw_leak.py` | `$BH/lens/aw/aw_leak.py` | `5729db45b7ee` | printed | none |  |
| `lens/aw/aw_equiv.py` | `$BH/lens/aw/aw_equiv.py` | `3ed0c48abf42` | printed | root |  |
| `lens/aw/aw_equiv_p0.py` | `$BH/lens/aw/aw_equiv_p0.py` | `06c86ddb3b2d` | printed | root; names |  |
| `lens/aw/p0_actconv.py` | `$BH/lens/aw/p0_actconv.py` | `1762d50085a3` | printed | root; names |  |
| `lens/aw/p0_datacheck.py` | `$BH/lens/aw/p0_datacheck.py` | `40cffeeb6761` | printed | root; names |  |
| `lens/aw/aw_dev3.py` | `$BH/lens/aw/aw_dev3_c2.py` | `cf217d3f39ed` | printed | root; names | c1 |
| `lens/aw/aw_pool.py` | `$BH/lens/aw/aw_pool_c3.py` | `e0a79dad6a5b` | printed | root; names | c2 (c3 adds the backbone prefix BBP, default pi05) |
| `lens/aw/e_check.py` | `$BH/lens/aw/e_check.py` | `fd32b511553a` | printed | none |  |
| `lens/aw/g_check.py` | `$BH/lens/aw/g_check.py` | `7d8e0c90ff9d` | printed | none |  |
| `lens/aw/h_check.py` | `$BH/lens/aw/h_check.py` | `fae0898cfd98` | printed | none |  |
| `lens/aw/i_check.py` | `$BH/lens/aw/i_check.py` | `8e1a92eafce6` | printed | none |  |
| `lens/aw/j_check.py` | `$BH/lens/aw/j_check.py` | `57f9571ff053` | printed | none |  |
| `lens/aw/m_check.py` | `$BH/lens/aw/m_check.py` | `9bc9d272f011` | printed | none | c1 (c2 reports instead of failing one smoke gate) |
| `lens/aw/posture.py` | `$BH/lens/aw/posture_c2.py` | `c09b0c73674b` | printed | root; names | c1 |
| `lens/aw/si_diag.py` | `$BH/lens/aw/si_diag_c2.py` | `6f49c12249e2` | printed | names | c1 |
| `lens/aw/switch_onset.py` | `$BH/lens/aw/switch_onset_c1.py` | `bb35af83b136` | uploaded | names |  |
| `lens/aw/trans_test.py` | `$BH/lens/aw/trans_test_c1.py` | `1ede75ed7d7b` | uploaded | root; names |  |
| `lens/aw/i_reach.py` | `$BH/lens/aw/i_reach_c1.py` | `ce3e5187919f` | uploaded | names |  |
| `lens/aw/ismk_look.py` | `$BH/lens/aw/ismk_look_c1.py` | `b78988169842` | uploaded | names |  |
| `lens/aw/freeze_anatomy.py` | `$BH/lens/aw/freeze_anatomy_c1.py` | `bc0b54a4179c` | printed | names |  |
| `lens/aw/cig_numerics.py` | `$BH/lens/aw/cig_numerics_c1.py` | `1c6db3616416` | printed | root; names |  |
| `lens/aw/freeze_probe.py` | `$BH/lens/aw/freeze_probe_c1.py` | `8b46f8da1d05` | printed | root; names |  |
| `lens/aw/g_fail.py` | `$BH/lens/aw/g_fail_c1.py` | `62fdad4369ab` | printed | names |  |
| `lens/aw/h0_trap.py` | `$BH/lens/aw/h0_trap_c1.py` | `07bf6d2b6cd9` | printed | root; names |  |
| `lens/aw/h0_side.py` | `$BH/lens/aw/h0_side_c1.py` | `a83b34762040` | uploaded | names |  |
| `lens/aw/fail_anatomy.py` | `$BH/lens/aw/fail_anatomy_c2.py` | `8dc547c5a7a3` | printed | names | c1 |
| `lens/aw/rlayout_check.py` | `$BH/lens/aw/rlayout_check_c2.py` | `7fa1c63a723c` | printed | names | c1 |
| `lens/aw/chain/lib.sh` | `$BH/lens/aw/chain/lib.sh` | `f22f1b2dd45f` | printed | root; names |  |
| `lens/aw/chain/lib2.sh` | `$BH/lens/aw/chain/lib2.sh` | `64ec7f1653ad` | printed | root; names |  |
| `lens/aw/chain/a1_dev2_select.sh` | `$BH/lens/aw/chain/a1_dev2_select.sh` | `db93f5be92ea` | printed | root; names |  |
| `lens/aw/chain/a2_select_test.sh` | `$BH/lens/aw/chain/a2_select_test.sh` | `e8429875ea13` | printed | root |  |
| `lens/aw/chain/b1_start_awc.sh` | `$BH/lens/aw/chain/b1_start_awc.sh` | `cfa474976ff1` | printed | root |  |
| `lens/aw/chain/b2_awc_awac.sh` | `$BH/lens/aw/chain/b2_awc_awac.sh` | `812e557f11d5` | printed | root |  |
| `lens/aw/chain/c_a_bc3s1.sh` | `$BH/lens/aw/chain/c_a_bc3s1.sh` | `7798e6d6687c` | printed | root |  |
| `lens/aw/chain/c_b_heads_awx3s1.sh` | `$BH/lens/aw/chain/c_b_heads_awx3s1.sh` | `114d755a2ae4` | printed | root |  |
| `lens/aw/chain/d_a_bc3p0.sh` | `$BH/lens/aw/chain/d_a_bc3p0.sh` | `4d96c3e22154` | printed | root |  |
| `lens/aw/chain/d_b_awx3p0.sh` | `$BH/lens/aw/chain/d_b_awx3p0.sh` | `549cfc06400a` | printed | root |  |
| `lens/aw/chain/e_lib.sh` | `$BH/lens/aw/chain/e_lib.sh` | `7d862ca25496` | printed | root; names |  |
| `lens/aw/chain/e_master.sh` | `$BH/lens/aw/chain/e_master.sh` | `6f0fa9d536d6` | printed | root; names |  |
| `lens/aw/chain/f_lib.sh` | `$BH/lens/aw/chain/f_lib.sh` | `9778f9a676fa` | printed | root; names |  |
| `lens/aw/chain/f_master.sh` | `$BH/lens/aw/chain/f_master_c2.sh` | `0a88a776aa52` | printed | root; names | c1 (c2 turns the self-check gate into a bf16 tolerance) |
| `lens/aw/chain/g_lib.sh` | `$BH/lens/aw/chain/g_lib.sh` | `c1efdb5217de` | printed | root; names |  |
| `lens/aw/chain/g_master.sh` | `$BH/lens/aw/chain/g_master.sh` | `6e1a924bcec5` | printed | root; names |  |
| `lens/aw/chain/h_lib.sh` | `$BH/lens/aw/chain/h_lib.sh` | `13cdac119078` | printed | root |  |
| `lens/aw/chain/h_master.sh` | `$BH/lens/aw/chain/h_master.sh` | `e68ca45ac2ad` | printed | root; names |  |
| `lens/aw/chain/i_lib.sh` | `$BH/lens/aw/chain/i_lib.sh` | `9ab8e466efa6` | printed | root; names |  |
| `lens/aw/chain/i_master.sh` | `$BH/lens/aw/chain/i_master.sh` | `1b4d4be501d6` | printed | root; names |  |
| `lens/aw/chain/j_lib.sh` | `$BH/lens/aw/chain/j_lib.sh` | `6a6164c94d9e` | printed | root; names |  |
| `lens/aw/chain/j_master.sh` | `$BH/lens/aw/chain/j_master.sh` | `981f6013e9a2` | printed | root; names |  |
| `lens/aw/chain/k_master.sh` | `$BH/lens/aw/chain/k_master.sh` | `439e4d6cc131` | printed | root; names |  |
| `lens/aw/chain/l_master.sh` | `$BH/lens/aw/chain/l_master.sh` | `5a3f5c6f3e70` | printed | root; names |  |
| `lens/aw/chain/m_master.sh` | `$BH/lens/aw/chain/m_master.sh` | `6019b518c562` | printed | root; names; one comment reworded |  |

md5 values are the first 12 hex digits of the md5 of the server file, before the changes above.
