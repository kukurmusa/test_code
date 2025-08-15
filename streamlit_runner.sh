#!/bin/bash
# Controller for running StreamlitNewsSentiment app
# Usage: ./StreamlitNewsSentiment.sh start|stop|status

# === CONFIGURE THESE ===
APP_DIR="/path/to/your/app"     # directory where app.py lives
APP="app.py"                   # your Streamlit app filename
CONDA_ENV="myenv"              # name of your conda environment
PORT=8501                      # port number

# internal
PIDFILE="$APP_DIR/StreamlitNewsSentiment.pid"
NAME="StreamlitNewsSentiment"

# helper: activate conda env
activate_conda() {
    # if conda isn't in PATH, source its init script
    if [ -z "$(command -v conda)" ]; then
        source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
        source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
    fi
    conda activate "$CONDA_ENV"
}

case "$1" in
  start)
    if [ -f "$PIDFILE" ]; then
      echo "$NAME already running (PID=$(cat $PIDFILE))."
      exit 1
    fi
    echo "Starting $NAME..."
    cd "$APP_DIR" || exit 1
    activate_conda
    # run in foreground, process named nicely
    exec -a "$NAME" streamlit run "$APP" --server.port=$PORT
    ;;
  stop)
    if [ ! -f "$PIDFILE" ]; then
      echo "No PID file, $NAME not tracked."
      exit 1
    fi
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo "Stopping $NAME (PID=$PID)..."
      kill "$PID"
      rm -f "$PIDFILE"
    else
      echo "Process $PID not running, cleaning up PID file."
      rm -f "$PIDFILE"
    fi
    ;;
  status)
    if [ -f "$PIDFILE" ]; then
      PID=$(cat "$PIDFILE")
      if kill -0 "$PID" 2>/dev/null; then
        echo "$NAME is running (PID=$PID)"
      else
        echo "$NAME PID file found, but process not running."
      fi
    else
      echo "$NAME not running."
    fi
    ;;
  *)
    echo "Usage: $0 {start|stop|status}"
    exit 1
    ;;
esac
