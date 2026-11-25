from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from collections import deque
import threading


app = Flask(__name__)
CORS(app)


# ============================================================
# CONFIGURATION
# ============================================================

# Grid demand is currently represented in kVA because
# the uploaded historical dataset provides kVA readings.
#
# LOW_SHED_THRESHOLD:
#     Low-priority load switches OFF.
#
# MEDIUM_SHED_THRESHOLD:
#     Medium-priority load also switches OFF.
#
# High-priority / critical load always remains ON automatically.

LOW_SHED_THRESHOLD = 2600.0
MEDIUM_SHED_THRESHOLD = 3000.0

MAX_HISTORY = 120

lock = threading.Lock()


# ============================================================
# HISTORICAL REFERENCE PROFILE
# ============================================================

# Representative profile based on the uploaded Aug data.
#
# Grid demand = 33 kV Incomer-1 kVA + 33 kV Incomer-2 kVA
#
# This data is ONLY used so the dashboard has a meaningful
# graph before live hardware readings start arriving.
#
# Once the first real /readings value is received,
# this reference profile is cleared automatically.

historical_reference = [

    ("07:00", 1889.0),
    ("08:00", 2012.0),
    ("09:00", 2188.0),
    ("10:00", 2736.0),
    ("11:00", 2386.0),
    ("12:00", 2622.0),
    ("13:00", 2485.0),
    ("14:00", 2505.0),
    ("15:00", 2767.0),
    ("16:00", 3090.0),
    ("17:00", 3003.0),
    ("18:00", 2697.0),
    ("19:00", 2690.0),
    ("20:00", 2657.0)

]


history = deque(maxlen=MAX_HISTORY)


for timestamp, grid_kva in historical_reference:

    history.append({
        "timestamp": timestamp,
        "grid_kva": grid_kva,
        "source": "historical"
    })


# ============================================================
# SYSTEM STATE
# ============================================================

system_state = {

    "grid_kva": historical_reference[-1][1],

    "relay_state": {

        "high": True,
        "medium": True,
        "low": True

    },

    "control_mode": "automatic",

    "load_status": "NORMAL",

    "last_updated":
        datetime.now().isoformat(timespec="seconds"),

    "data_source": "historical",

    "live_started": False

}


# ============================================================
# AUTOMATIC LOAD SHEDDING
# ============================================================

def calculate_automatic_relays(grid_kva):

    relay_state = {

        "high": True,
        "medium": True,
        "low": True

    }


    # --------------------------------------------------------
    # PEAK DEMAND
    #
    # Critical load only.
    # --------------------------------------------------------

    if grid_kva >= MEDIUM_SHED_THRESHOLD:

        relay_state["high"] = True

        relay_state["medium"] = False

        relay_state["low"] = False

        status = "PEAK"


    # --------------------------------------------------------
    # HIGH DEMAND
    #
    # Low priority is shed first.
    # --------------------------------------------------------

    elif grid_kva >= LOW_SHED_THRESHOLD:

        relay_state["high"] = True

        relay_state["medium"] = True

        relay_state["low"] = False

        status = "HIGH"


    # --------------------------------------------------------
    # NORMAL DEMAND
    #
    # All loads remain active.
    # --------------------------------------------------------

    else:

        relay_state["high"] = True

        relay_state["medium"] = True

        relay_state["low"] = True

        status = "NORMAL"


    return relay_state, status


# ============================================================
# STATUS
# ============================================================

def calculate_status(grid_kva):

    if grid_kva >= MEDIUM_SHED_THRESHOLD:

        return "PEAK"


    if grid_kva >= LOW_SHED_THRESHOLD:

        return "HIGH"


    return "NORMAL"


# ============================================================
# UPDATE AUTOMATIC CONTROL
# ============================================================

def update_automatic_control():

    if system_state["control_mode"] != "automatic":

        return


    relay_state, status = calculate_automatic_relays(

        system_state["grid_kva"]

    )


    system_state["relay_state"] = relay_state

    system_state["load_status"] = status


# Apply control logic to initial historical value.

update_automatic_control()


# ============================================================
# ADD HISTORY
# ============================================================

def add_history(grid_kva, source="live"):

    history.append({

        "timestamp":
            datetime.now().strftime("%H:%M:%S"),

        "grid_kva":
            round(float(grid_kva), 2),

        "source":
            source

    })


# ============================================================
# ROOT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "status": "ok",

        "message":
            "MMMUT Priority Based Energy Management System",

        "grid_unit":
            "kVA",

        "control_mode":
            system_state["control_mode"],

        "data_source":
            system_state["data_source"]

    })


# ============================================================
# RECEIVE LIVE GRID READING
# ============================================================

@app.route("/readings", methods=["POST"])
def receive_reading():

    data = request.get_json(silent=True) or {}


    # Accept either grid_kva or grid_kw temporarily
    # so older ESP32/frontend code does not immediately break.

    value = data.get("grid_kva")

    if value is None:

        value = data.get("grid_kw")


    if value is None:

        return jsonify({

            "error":
                "grid_kva is required"

        }), 400


    try:

        grid_kva = float(value)

    except (TypeError, ValueError):

        return jsonify({

            "error":
                "grid_kva must be numeric"

        }), 400


    if grid_kva < 0:

        return jsonify({

            "error":
                "grid_kva cannot be negative"

        }), 400


    with lock:


        # ----------------------------------------------------
        # FIRST LIVE READING
        #
        # Remove historical reference graph.
        # ----------------------------------------------------

        if not system_state["live_started"]:

            history.clear()

            system_state["live_started"] = True


        system_state["grid_kva"] = grid_kva

        system_state["data_source"] = "live"


        system_state["last_updated"] = (

            datetime.now()
            .isoformat(timespec="seconds")

        )


        add_history(
            grid_kva,
            "live"
        )


        if system_state["control_mode"] == "automatic":

            update_automatic_control()

        else:

            system_state["load_status"] = (

                calculate_status(grid_kva)

            )


        response = {

            "message":
                "Grid reading received",

            "grid_kva":
                grid_kva,

            "load_status":
                system_state["load_status"],

            "control_mode":
                system_state["control_mode"],

            "relay_state":
                system_state["relay_state"]

        }


    return jsonify(response)


# ============================================================
# LATEST READING
# ============================================================

@app.route("/latest", methods=["GET"])
def latest():

    with lock:

        return jsonify({

            "latest": {

                "grid_kva":
                    system_state["grid_kva"],

                "timestamp":
                    system_state["last_updated"],

                "load_status":
                    system_state["load_status"],

                "data_source":
                    system_state["data_source"]

            },

            "history": {

                "timestamps": [

                    item["timestamp"]
                    for item in history

                ],

                "grid_kva": [

                    item["grid_kva"]
                    for item in history

                ],

                "sources": [

                    item["source"]
                    for item in history

                ]

            }

        })


# ============================================================
# SYSTEM COMMANDS / DASHBOARD STATE
# ============================================================

@app.route("/commands", methods=["GET"])
def commands():

    with lock:

        current_history = list(history)


        return jsonify({

            "generated_at":

                datetime.now()
                .isoformat(timespec="seconds"),


            "grid_kva":

                system_state["grid_kva"],


            "grid_unit":

                "kVA",


            "control_mode":

                system_state["control_mode"],


            "load_status":

                system_state["load_status"],


            "data_source":

                system_state["data_source"],


            "relay_state":

                system_state["relay_state"],


            "thresholds": {

                "low_shed_kva":
                    LOW_SHED_THRESHOLD,

                "medium_shed_kva":
                    MEDIUM_SHED_THRESHOLD

            },


            "history": {

                "timestamps": [

                    item["timestamp"]
                    for item in current_history

                ],

                "grid_kva": [

                    item["grid_kva"]
                    for item in current_history

                ],

                "sources": [

                    item["source"]
                    for item in current_history

                ]

            }

        })


# ============================================================
# MANUAL RELAY CONTROL
# ============================================================

@app.route("/relay", methods=["POST"])
def relay_control():

    data = request.get_json(silent=True) or {}


    priority = str(

        data.get(
            "priority",
            ""
        )

    ).lower()


    state = data.get("state")


    if priority not in [

        "high",
        "medium",
        "low"

    ]:

        return jsonify({

            "error":
                "priority must be high, medium or low"

        }), 400


    if not isinstance(state, bool):

        return jsonify({

            "error":
                "state must be true or false"

        }), 400


    # --------------------------------------------------------
    # CRITICAL LOAD PROTECTION
    # --------------------------------------------------------

    if priority == "high" and state is False:

        return jsonify({

            "error":
                "High priority critical load cannot be switched OFF"

        }), 400


    with lock:

        system_state["control_mode"] = "manual"


        system_state[
            "relay_state"
        ][priority] = state


        return jsonify({

            "message":
                f"{priority} priority relay updated",

            "control_mode":
                system_state["control_mode"],

            "relay_state":
                system_state["relay_state"]

        })


# ============================================================
# RESUME AUTOMATIC MODE
# ============================================================

@app.route("/auto", methods=["POST"])
def automatic_control():

    with lock:

        system_state["control_mode"] = "automatic"


        update_automatic_control()


        return jsonify({

            "message":
                "Automatic priority control resumed",

            "control_mode":
                system_state["control_mode"],

            "load_status":
                system_state["load_status"],

            "relay_state":
                system_state["relay_state"]

        })


# ============================================================
# UPDATE THRESHOLDS
# ============================================================

@app.route("/thresholds", methods=["POST"])
def update_thresholds():

    global LOW_SHED_THRESHOLD

    global MEDIUM_SHED_THRESHOLD


    data = request.get_json(silent=True) or {}


    try:

        low_threshold = float(

            data.get(

                "low_shed_kva",

                LOW_SHED_THRESHOLD

            )

        )


        medium_threshold = float(

            data.get(

                "medium_shed_kva",

                MEDIUM_SHED_THRESHOLD

            )

        )


    except (TypeError, ValueError):

        return jsonify({

            "error":
                "Thresholds must be numeric"

        }), 400


    if (

        low_threshold <= 0

        or

        medium_threshold <= 0

    ):

        return jsonify({

            "error":
                "Thresholds must be greater than zero"

        }), 400


    if low_threshold >= medium_threshold:

        return jsonify({

            "error":
                "Low priority threshold must be below peak threshold"

        }), 400


    with lock:

        LOW_SHED_THRESHOLD = low_threshold

        MEDIUM_SHED_THRESHOLD = medium_threshold


        if system_state["control_mode"] == "automatic":

            update_automatic_control()


    return jsonify({

        "message":
            "Thresholds updated",

        "thresholds": {

            "low_shed_kva":
                LOW_SHED_THRESHOLD,

            "medium_shed_kva":
                MEDIUM_SHED_THRESHOLD

        }

    })


# ============================================================
# SIMULATION
# ============================================================

@app.route("/simulate", methods=["POST"])
def simulate():

    data = request.get_json(silent=True) or {}


    value = data.get("grid_kva")


    if value is None:

        return jsonify({

            "error":
                "grid_kva is required"

        }), 400


    try:

        grid_kva = float(value)

    except (TypeError, ValueError):

        return jsonify({

            "error":
                "grid_kva must be numeric"

        }), 400


    if grid_kva < 0:

        return jsonify({

            "error":
                "grid_kva cannot be negative"

        }), 400


    with lock:

        system_state["grid_kva"] = grid_kva


        system_state["last_updated"] = (

            datetime.now()
            .isoformat(timespec="seconds")

        )


        system_state["data_source"] = "simulation"


        add_history(
            grid_kva,
            "simulation"
        )


        if system_state["control_mode"] == "automatic":

            update_automatic_control()

        else:

            system_state["load_status"] = (

                calculate_status(grid_kva)

            )


        return jsonify({

            "grid_kva":
                grid_kva,

            "control_mode":
                system_state["control_mode"],

            "load_status":
                system_state["load_status"],

            "relay_state":
                system_state["relay_state"]

        })


# ============================================================
# RESET TO HISTORICAL PROFILE
# ============================================================

@app.route("/reset-demo", methods=["POST"])
def reset_demo():

    with lock:

        history.clear()


        for timestamp, grid_kva in historical_reference:

            history.append({

                "timestamp":
                    timestamp,

                "grid_kva":
                    grid_kva,

                "source":
                    "historical"

            })


        system_state["grid_kva"] = (

            historical_reference[-1][1]

        )


        system_state["data_source"] = "historical"

        system_state["live_started"] = False

        system_state["control_mode"] = "automatic"


        system_state["last_updated"] = (

            datetime.now()
            .isoformat(timespec="seconds")

        )


        update_automatic_control()


        return jsonify({

            "message":
                "Historical reference profile restored",

            "grid_kva":
                system_state["grid_kva"],

            "relay_state":
                system_state["relay_state"]

        })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "===================================================="
    )

    print(
        " MMMUT PRIORITY BASED ENERGY MANAGEMENT SYSTEM"
    )

    print(
        "===================================================="
    )

    print()

    print(
        "Grid measurement unit: kVA"
    )

    print()

    print(
        f"Grid < {LOW_SHED_THRESHOLD:.0f} kVA"
    )

    print(
        "  HIGH   : ON"
    )

    print(
        "  MEDIUM : ON"
    )

    print(
        "  LOW    : ON"
    )

    print()

    print(
        f"Grid >= {LOW_SHED_THRESHOLD:.0f} kVA"
    )

    print(
        "  HIGH   : ON"
    )

    print(
        "  MEDIUM : ON"
    )

    print(
        "  LOW    : OFF"
    )

    print()

    print(
        f"Grid >= {MEDIUM_SHED_THRESHOLD:.0f} kVA"
    )

    print(
        "  HIGH   : ON"
    )

    print(
        "  MEDIUM : OFF"
    )

    print(
        "  LOW    : OFF"
    )

    print()

    print(
        "Critical HIGH priority load is always protected."
    )

    print()

    app.run(

        host="0.0.0.0",

        port=5005,

        debug=True

    )