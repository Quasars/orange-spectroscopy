if [ "$RUN_PYLINT" ]; then
    cd $TRAVIS_BUILD_DIR
    pip install pylint
    cp pylintrc ~/.pylintrc
    .travis/check_pylint_diff
    EXIT_CODE=$?
    echo "Lint check returned ${EXIT_CODE}"
    return ${EXIT_CODE}
else
    # Screen must be 24bpp lest pyqt5 crashes, see pytest-dev/pytest-qt/35
    export XVFBARGS="-screen 0 1280x1024x24"
    catchsegv xvfb-run -a -s "$XVFBARGS" coverage run -m unittest orangecontrib.spectroscopy
    (($? != 0)) && { printf '%s\n' "Command exited with non-zero"; exit 1; }
fi

if [ "$UPLOAD_COVERAGE" ]; then
    cp $TRAVIS_BUILD_DIR/codecov.yml .
    codecov
fi
