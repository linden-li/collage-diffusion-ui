#!/bin/bash

if [[ "$1" == "frontend" ]]; then
  echo "Killing npm + node processes"
  ps -fw -u $USER | grep npm | grep -v grep | awk '{print $2}' | xargs kill
  ps -fw -u $USER | grep node | grep -v grep | awk '{print $2}' | xargs kill
  ps -fw -u $USER | grep serve | grep -v grep | awk '{print $2}' | xargs kill
fi 

if [[ "$1" == "backend" ]]; then
  echo "Killing ray + python processes"
  ray stop --force
  ps -fw -u $USER | grep python | grep -v grep | awk '{print $2}' | xargs kill
fi 
