while true; do
    ros2 service call /reset_world std_srvs/srv/Empty "{}"
    sleep 0.1
    echo "Resetting world"
done