#!/bin/bash
exec /usr/bin/time -f "\nExecution Time: %E" python main.py
echo -e "\a"
