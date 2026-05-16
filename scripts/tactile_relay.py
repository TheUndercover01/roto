#!/usr/bin/env python3
"""Run this in the TouchLab container.

Subscribes to /touchlab_driver/calibrated (Float64MultiArrayStamped, TouchLab container only)
and republishes just the data array as /touchlab_driver/calibrated_flat (std_msgs/Float64MultiArray),
which is accessible from any container without the touchlab_ros package.

Usage (in TouchLab container):
    python tactile_relay.py
"""

import rospy
from touchlab_msgs.msg import Float64MultiArrayStamped
from std_msgs.msg import Float64MultiArray

_pub = None

def _callback(msg):
    out = Float64MultiArray()
    out.layout = msg.multi_array.layout
    out.data   = msg.multi_array.data
    _pub.publish(out)

if __name__ == "__main__":
    rospy.init_node("tactile_relay", anonymous=False)
    _pub = rospy.Publisher("/touchlab_driver/calibrated_flat", Float64MultiArray, queue_size=1)
    rospy.Subscriber("/touchlab_driver/calibrated", Float64MultiArrayStamped, _callback, queue_size=1)
    rospy.loginfo("tactile_relay: /touchlab_driver/calibrated → /touchlab_driver/calibrated_flat")
    rospy.spin()
