"""Built-in tool-calling eval (BFCL-style, "FC" mode): OpenAI `tools` definitions, one user turn, expected calls.

Categories (as in the Berkeley Function-Calling Leaderboard):
  simple       one tool offered, call it with the right arguments
  multiple     several tools offered, pick the right one
  parallel     the request needs two calls in the same reply
  irrelevance  tools are offered but none fits: the model must answer WITHOUT calling a tool

An expected argument is a list of acceptable values; None in that list means the argument may be left out.
Values are compared case-insensitively, numbers numerically, lists as unordered sets. An argument that is not in
the tool's schema fails the case (a made-up parameter would break a real agent).

These are written for this tool (no download, no licence question) and kept small and unambiguous: the point is to
catch chat-template / --jinja / tool-parser problems and weak argument filling, not to rank frontier models.
"""
from __future__ import annotations


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


S = {"type": "string"}
I = {"type": "integer"}
N = {"type": "number"}

TOOLS = {
    "get_weather": _fn("get_weather", "Current weather for a city.",
                       {"city": S, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    "convert_currency": _fn("convert_currency", "Convert an amount between currencies (ISO codes).",
                            {"amount": N, "from_currency": S, "to_currency": S}, ["amount", "from_currency", "to_currency"]),
    "set_thermostat": _fn("set_thermostat", "Set the target temperature (Celsius) of a room's thermostat.",
                          {"room": S, "temperature": N, "mode": {"type": "string", "enum": ["heat", "cool", "auto"]}},
                          ["room", "temperature"]),
    "turn_on_light": _fn("turn_on_light", "Turn on a Home Assistant light entity, optionally at a brightness (0-100).",
                         {"entity_id": S, "brightness": I}, ["entity_id"]),
    "search_web": _fn("search_web", "Search the web.", {"query": S, "max_results": I}, ["query"]),
    "create_calendar_event": _fn("create_calendar_event", "Create a calendar event.",
                                 {"title": S, "date": {"type": "string", "description": "YYYY-MM-DD"},
                                  "time": {"type": "string", "description": "HH:MM, 24-hour"},
                                  "duration_minutes": I}, ["title", "date"]),
    "send_email": _fn("send_email", "Send an email.", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
    "calculate_mortgage": _fn("calculate_mortgage", "Monthly repayment of a loan.",
                              {"principal": N, "annual_rate_pct": N, "years": I}, ["principal", "annual_rate_pct", "years"]),
    "get_stock_price": _fn("get_stock_price", "Latest price of a stock by ticker symbol.", {"ticker": S}, ["ticker"]),
    "add_todo": _fn("add_todo", "Add an item to the to-do list.",
                    {"text": S, "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                     "tags": {"type": "array", "items": S}}, ["text"]),
    "read_file": _fn("read_file", "Read a text file.", {"path": S}, ["path"]),
    "list_directory": _fn("list_directory", "List the files in a directory.", {"path": S}, ["path"]),
    "play_music": _fn("play_music", "Play music on a speaker.", {"artist": S, "song": S, "room": S}),
    "get_distance": _fn("get_distance", "Travel distance and time between two places.",
                        {"origin": S, "destination": S, "mode": {"type": "string", "enum": ["driving", "walking", "transit"]}},
                        ["origin", "destination"]),
    "translate_text": _fn("translate_text", "Translate text into another language.",
                          {"text": S, "target_language": S}, ["text", "target_language"]),
    "set_timer": _fn("set_timer", "Start a countdown timer.", {"minutes": I, "label": S}, ["minutes"]),
}


def _c(cid, cat, tools, prompt, expect):
    return {"id": cid, "category": cat, "tools": tools, "prompt": prompt, "expect": expect}


def call(name, **args):
    return {"name": name, "args": args}


CASES = [
    # ---- simple -------------------------------------------------------------------------------------------------
    _c("simple-01", "simple", ["get_weather"], "What's the weather in Melbourne right now? Use Celsius.",
       [call("get_weather", city=["Melbourne", "Melbourne, Australia", "Melbourne, VIC", "Melbourne, AU"],
             unit=["celsius"])]),
    _c("simple-02", "simple", ["convert_currency"], "Convert 250 US dollars to euros.",
       [call("convert_currency", amount=[250], from_currency=["USD"], to_currency=["EUR"])]),
    _c("simple-03", "simple", ["set_thermostat"], "Set the bedroom thermostat to 21 degrees.",
       [call("set_thermostat", room=["bedroom"], temperature=[21], mode=[None, "auto", "heat"])]),
    _c("simple-04", "simple", ["turn_on_light"], "Turn on light.kitchen_main at 40% brightness.",
       [call("turn_on_light", entity_id=["light.kitchen_main"], brightness=[40])]),
    _c("simple-05", "simple", ["create_calendar_event"],
       "Add 'Dentist' to my calendar on 2026-12-03 at 14:30 for 45 minutes.",
       [call("create_calendar_event", title=["Dentist"], date=["2026-12-03"], time=["14:30"], duration_minutes=[45])]),
    _c("simple-06", "simple", ["calculate_mortgage"],
       "What's the monthly repayment on a 600000 loan at 6.1% over 30 years?",
       [call("calculate_mortgage", principal=[600000], annual_rate_pct=[6.1], years=[30])]),
    _c("simple-07", "simple", ["add_todo"], "Add 'renew car registration' to my to-do list as high priority, tagged car.",
       [call("add_todo", text=["renew car registration", "Renew car registration"], priority=["high"],
             tags=[["car"], None])]),
    _c("simple-08", "simple", ["set_timer"], "Start a 12 minute timer called pasta.",
       [call("set_timer", minutes=[12], label=["pasta", "Pasta"])]),
    _c("simple-09", "simple", ["get_stock_price"], "What's NVIDIA trading at? The ticker is NVDA.",
       [call("get_stock_price", ticker=["NVDA"])]),
    _c("simple-10", "simple", ["translate_text"], "Translate 'where is the train station' into Japanese.",
       [call("translate_text", text=["where is the train station", "Where is the train station",
                                     "Where is the train station?", "where is the train station?"],
             target_language=["Japanese", "ja", "jp"])]),
    # ---- multiple -----------------------------------------------------------------------------------------------
    _c("multiple-01", "multiple", ["get_weather", "get_distance", "search_web"],
       "How far is it to drive from Geelong to Ballarat?",
       [call("get_distance", origin=["Geelong"], destination=["Ballarat"], mode=["driving", None])]),
    _c("multiple-02", "multiple", ["read_file", "list_directory", "send_email"], "Show me the contents of /etc/hostname.",
       [call("read_file", path=["/etc/hostname"])]),
    _c("multiple-03", "multiple", ["read_file", "list_directory", "send_email"], "What files are in /var/log?",
       [call("list_directory", path=["/var/log", "/var/log/"])]),
    _c("multiple-04", "multiple", ["play_music", "turn_on_light", "set_timer"], "Play Thunderstruck by AC/DC in the garage.",
       [call("play_music", artist=["AC/DC", "ACDC", "AC DC"], song=["Thunderstruck"], room=["garage"])]),
    _c("multiple-05", "multiple", ["get_stock_price", "convert_currency", "search_web"],
       "How many Japanese yen is 80 Australian dollars?",
       [call("convert_currency", amount=[80], from_currency=["AUD"], to_currency=["JPY"])]),
    _c("multiple-06", "multiple", ["send_email", "create_calendar_event", "add_todo"],
       "Email sam@example.com with the subject 'Lunch' and say 'Running 10 minutes late.'",
       [call("send_email", to=["sam@example.com"], subject=["Lunch"],
             body=["Running 10 minutes late.", "Running 10 minutes late", "running 10 minutes late."])]),
    _c("multiple-07", "multiple", ["set_thermostat", "turn_on_light", "get_weather"],
       "Cool the living room down to 23 degrees.",
       [call("set_thermostat", room=["living room", "living_room", "livingroom"], temperature=[23], mode=["cool", None])]),
    _c("multiple-08", "multiple", ["search_web", "read_file", "get_weather"],
       "Search the web for 'llama.cpp n-cpu-moe flag' and give me 5 results.",
       [call("search_web", query=["llama.cpp n-cpu-moe flag"], max_results=[5])]),
    _c("multiple-09", "multiple", ["set_timer", "create_calendar_event", "add_todo"],
       "Remind me in 25 minutes to take the washing out (just a timer is fine).",
       [call("set_timer", minutes=[25], label=[None, "take the washing out", "washing", "Take the washing out"])]),
    _c("multiple-10", "multiple", ["translate_text", "search_web", "send_email"], "How do you say 'thank you' in German?",
       [call("translate_text", text=["thank you", "Thank you"], target_language=["German", "de"])]),
    # ---- parallel -----------------------------------------------------------------------------------------------
    _c("parallel-01", "parallel", ["get_weather"], "What's the weather in Sydney and in Perth? Celsius please.",
       [call("get_weather", city=["Sydney", "Sydney, Australia", "Sydney, NSW"], unit=["celsius"]),
        call("get_weather", city=["Perth", "Perth, Australia", "Perth, WA"], unit=["celsius"])]),
    _c("parallel-02", "parallel", ["turn_on_light", "play_music"],
       "Turn on light.lounge at full brightness and play some Coldplay in the lounge.",
       [call("turn_on_light", entity_id=["light.lounge"], brightness=[100, None]),
        call("play_music", artist=["Coldplay"], room=["lounge"], song=[None])]),
    _c("parallel-03", "parallel", ["get_stock_price"], "Get me the prices for AAPL and MSFT.",
       [call("get_stock_price", ticker=["AAPL"]), call("get_stock_price", ticker=["MSFT"])]),
    _c("parallel-04", "parallel", ["set_timer", "add_todo"],
       "Set a 5 minute timer for tea, and add 'buy milk' to my to-do list.",
       [call("set_timer", minutes=[5], label=["tea", "Tea"]),
        call("add_todo", text=["buy milk", "Buy milk"], priority=[None], tags=[None])]),
    _c("parallel-05", "parallel", ["convert_currency"], "Convert 100 EUR to USD and 100 EUR to GBP.",
       [call("convert_currency", amount=[100], from_currency=["EUR"], to_currency=["USD"]),
        call("convert_currency", amount=[100], from_currency=["EUR"], to_currency=["GBP"])]),
    _c("parallel-06", "parallel", ["read_file"], "Read both /etc/hosts and /etc/fstab.",
       [call("read_file", path=["/etc/hosts"]), call("read_file", path=["/etc/fstab"])]),
    # ---- irrelevance --------------------------------------------------------------------------------------------
    _c("irrelevance-01", "irrelevance", ["get_weather", "convert_currency"], "What is 17 multiplied by 23?", []),
    _c("irrelevance-02", "irrelevance", ["send_email", "read_file"], "Who wrote the novel 'Pride and Prejudice'?", []),
    _c("irrelevance-03", "irrelevance", ["set_thermostat", "turn_on_light"], "Tell me a short joke about computers.", []),
    _c("irrelevance-04", "irrelevance", ["get_stock_price"], "Explain in one sentence what a stock dividend is.", []),
    _c("irrelevance-05", "irrelevance", ["get_distance", "play_music"], "What is the capital city of Canada?", []),
    _c("irrelevance-06", "irrelevance", ["create_calendar_event", "set_timer"], "How many days are there in a leap year?", []),
]

SYSTEM = ("You are a helpful assistant with access to tools. When a request needs a tool, call it with exactly the "
          "arguments the user gave. When a request needs several calls, make all of them in the same reply. If no "
          "tool fits, answer normally without calling a tool.")


def build_items():
    """Items in the shape the eval runner expects."""
    items = []
    for c in CASES:
        items.append({"id": f"tool-{c['id']}", "category": c["category"], "prompt": c["prompt"],
                      "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": c["prompt"]}],
                      "tools": [TOOLS[t] for t in c["tools"]], "expect": c["expect"]})
    return items
