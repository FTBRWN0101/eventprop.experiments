@echo off
REM Scores every saved checkpoint of one SNN cell, to ask whether it should have
REM stopped training earlier. Inference only, no training, so it is minutes rather
REM than a rerun.
REM
REM NOTE on reading the output. Checkpoints trained without --holdout-val had the
REM 2017-2019 rows in their training data, so for those the val column is in-sample
REM and cannot be used to choose an epoch. Read the test column as a DIAGNOSTIC:
REM if it peaks before the final epoch, the network is overfitting and early stopping
REM is worth building. Choosing the epoch on test would be the selection-on-test
REM problem this whole exercise exists to fix.
REM
REM usage: run_epoch_curve.bat --encoding delta --input-window 45 --seed 1 --readout-window
call "%~dp0bootstrap.bat"
call "%~dp0run_in_mlgenn.bat" "%REPO%\src\experiments\tools\epoch_curve.py" %*
exit /b %errorlevel%
