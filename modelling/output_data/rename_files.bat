@echo off
setlocal enabledelayedexpansion

set "folder_path=%CD%"

for %%A in ("%folder_path%\*") do (
    if exist "%%A" (
        set "filename=%%~nxA"
        set "extension=%%~xA"

        if "!extension!"=="" (
            set "new_filename=!filename!.csv"
            set "new_filepath=!folder_path!\!new_filename!"

            if not "!filename!"=="!new_filename!" (
                ren "%%A" "!new_filename!"
                echo Renamed '!filename!' to '!new_filename!'
            )
        )
    )
)

endlocal
