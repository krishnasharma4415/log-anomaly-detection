"""
Log parsing regex patterns for different log types
"""
import re

REGEX_PATTERNS = {
    "Android": re.compile(
        r"^(?P<Date>\d{2}-\d{2})\s+(?P<Time>\d{2}:\d{2}:\d{2}\.\d+)\s+(?P<Pid>\d+)\s+(?P<Tid>\d+)\s+(?P<Level>[VDIWEF])\s+(?P<Component>\S+)\s+(?P<Content>.+)$"
    ),
    "Apache": re.compile(
        r"^\[(?P<Date>[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+\d{4})\]\s+\[(?P<Level>\w+)\]\s+(?P<Content>.+)$"
    ),
    "BGL": re.compile(
        r"^(?P<Label>[^,]+),(?P<Timestamp>\d+),(?P<Date>[\d.]+),(?P<Node>[^,]+),(?P<Time>[^,]+),(?P<NodeRepeat>[^,]+),(?P<Type>[^,]+),(?P<Component>[^,]+),(?P<Level>[^,]+),(?P<Content>.+)$"
    ),
    "Hadoop": re.compile(
        r"^(?P<Date>\d{4}-\d{2}-\d{2}),\"(?P<Time>[\d:,]+)\",(?P<Level>\w+),(?P<Process>[^,]+),(?P<Component>[^,]+),(?P<Content>.+)$"
    ),
    "HDFS": re.compile(
        r"^(?P<Date>\d+),(?P<Time>\d+),(?P<Pid>\d+),(?P<Level>\w+),(?P<Component>[^,]+),(?P<Content>.+)$"
    ),
    "HealthApp": re.compile(
        r"^(?P<Time>\d{8}-\d{2}:\d{2}:\d{2}:\d+),(?P<Component>[^,]+),(?P<Pid>\d+),(?P<Content>.+)$"
    ),
    "HPC": re.compile(
        r"^(?P<LogId>\d+),(?P<Node>[^,]+),(?P<Component>[^,]+),(?P<State>[^,]+),(?P<Time>\d+),(?P<Flag>\d+),(?P<Content>.+)$"
    ),
    "Linux": re.compile(
        r"^(?P<Month>[A-Za-z]+)\s+(?P<Date>\d+)\s+(?P<Time>\d{2}:\d{2}:\d{2})\s+(?P<Component>[^(]+)\((?P<PID>\d+)\):\s+(?P<Content>.+)$"
    ),
    "Mac": re.compile(
        r"^(?P<Month>[A-Za-z]+)\s+(?P<Date>\d+)\s+(?P<Time>\d{2}:\d{2}:\d{2})\s+(?P<User>[^,]+),(?P<Component>[^,]+),(?P<PID>\d+),(?P<Address>[^,]*),(?P<Content>.+)$"
    ),
    "OpenSSH": re.compile(
        r"^(?P<Month>[A-Za-z]+)\s+(?P<Day>\d+)\s+(?P<Time>\d{2}:\d{2}:\d{2})\s+(?P<Component>[^,]+)\[(?P<Pid>\d+)\]:\s+(?P<Content>.+)$"
    ),
    "OpenStack": re.compile(
        r"^(?P<Logrecord>[^,]+),(?P<Date>\d{4}-\d{2}-\d{2}),(?P<Time>[\d:.]+),(?P<Pid>\d+),(?P<Level>\w+),(?P<Component>[^,]+),(?P<ADDR>[^,]+),\"(?P<Content>.+)\"$"
    ),
    "Proxifier": re.compile(
        r"^(?P<Time>\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}),(?P<Program>[^,]+),(?P<Content>.+)$"
    ),
    "Spark": re.compile(
        r"^(?P<Date>\d{2}/\d{2}/\d{2}),(?P<Time>\d{2}:\d{2}:\d{2}),(?P<Level>\w+),(?P<Component>[^,]+),\"(?P<Content>.+)\"$"
    ),
    "Thunderbird": re.compile(
        r"^(?P<Label>[^,]+),(?P<Timestamp>\d+),(?P<Date>[\d.]+),(?P<User>[^,]+),(?P<Month>\w+),(?P<Day>\d+),(?P<Time>\d{2}:\d{2}:\d{2}),(?P<Location>[^,]+),(?P<Component>[^(]+)\((?P<PID>[^)]+)\),(?P<Content>.+)$"
    ),
    "Windows": re.compile(
        r"^(?P<Date>\d{4}-\d{2}-\d{2}),(?P<Time>\d{2}:\d{2}:\d{2}),(?P<Level>\w+),(?P<Component>[^,]+),(?P<Content>.+)$"
    ),
    "Zookeeper": re.compile(
        r"^(?P<Date>\d{4}-\d{2}-\d{2}),\"(?P<Time>[\d:,]+)\",(?P<Level>\w+),(?P<Node>[^,]+),(?P<Component>[^,]+),(?P<Id>\d+),(?P<Content>.+)$"
    ),
}