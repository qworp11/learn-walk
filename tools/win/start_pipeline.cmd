@echo off
rem Автозапуск конвейера обучения при входе в Windows
wsl.exe -d Ubuntu-22.04 -u root -- bash -c "bash '/mnt/g/LEARN WALK/tools/wsl/boot_pipeline.sh' s2_walk s2_walk_run_a s3_run s3_sprint s4_jump s6_fall s6_fall2 s8_dance"
