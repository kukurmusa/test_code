#!/bin/bash
# Controller for running StreamlitNewsSentiment app
# Usage: ./StreamlitNewsSentiment.sh start|stop|status|restart

# === CONFIGURE THESE ===
APP_DIR="/path/to/your/app"     # directory where app.py lives
APP="app.py"                    # your Streamlit app filename
CONDA_ENV="myenv"               # conda environment name
PORT=8501                       # port number

# Internal
PIDFILE="$APP_DIR/StreamlitNewsSentiment.pid"
NAME="StreamlitNewsSentiment"

activate_conda() {
  # Load conda if not already available
  if ! command -v conda >/dev/null 2>&1; then
    # adjust these paths if your conda lives elsewhere
    [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] && source "$HOME/miniconda3/etc/profile.d/conda.sh"
    [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ] && source "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
  conda activate "$CONDA_ENV"
}

start_app() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "$NAME already running (PID=$(cat "$PIDFILE"))."
    exit 1
  fi

  echo "Starting $NAME..."
  cd "$APP_DIR" || { echo "Cannot cd to $APP_DIR"; exit 1; }
  activate_conda

  # Start Streamlit with a custom process name, capture PID, then wait (foreground behaviour)
  bash -c "exec -a '$NAME' streamlit run '$APP' --server.port=$PORT" &
  CHILD_PID=$!

  # Write PID and set cleanup traps
  echo "$CHILD_PID" > "$PIDFILE"

  # Ensure PID file is removed when the process exits or if you Ctrl+C this script
  trap 'rm -f "$PIDFILE"' INT TERM EXIT

  # Keep the script attached to the child (so it feels like foreground)
  wait "$CHILD_PID"
  EXIT_CODE=$?
  rm -f "$PIDFILE"
  exit $EXIT_CODE
}

stop_app() {
  if [ ! -f "$PIDFILE" ]; then
    echo "No PID file; $NAME not tracked."
    exit 1
  fi
  PID=$(cat "$PIDFILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping $NAME (PID=$PID)..."
    kill "$PID"
    # polite wait, then force if needed
    for i in {1..10}; do
      kill -0 "$PID" 2>/dev/null || break
      sleep 0.3
    end
    if kill -0 "$PID" 2>/dev/null; then
      echo "Force killing..."
      kill -9 "$PID"
    fi
  else
    echo "Process $PID not running."
  fi
  rm -f "$PIDFILE"
}

status_app() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "$NAME is running (PID=$(cat "$PIDFILE"))"
  else
    # Try to find by name as fallback
    PIDS=$(pgrep -f "$NAME")
    if [ -n "$PIDS" ]; then
      echo "$NAME seems running (PIDs: $PIDS) but PID file missing."
    else
      echo "$NAME not running."
    fi
  fi
}

case "$1" in
  start)   start_app ;;
  stop)    stop_app ;;
  restart) stop_app; start_app ;;
  status)  status_app ;;
  *) echo "Usage: $0 {start|stop|restart|status}"; exit 1 ;;
esac
Why this works:
