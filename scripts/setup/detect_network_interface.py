#!/usr/bin/env python3
import psutil
import sys

def get_network_interfaces():
    interfaces = []
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == 2:  # IPv4
                interfaces.append({
                    'name': interface,
                    'ip': addr.address,
                    'netmask': addr.netmask
                })
    return interfaces

if __name__ == "__main__":
    print("Available network interfaces:")
    interfaces = get_network_interfaces()
    for i, interface in enumerate(interfaces):
        print(f"{i+1}. {interface['name']} - {interface['ip']}")
    
    if interfaces:
        print(f"\nRecommended interface: {interfaces[0]['name']}")
        print(f"Update config/live_config.yaml with: interface: '{interfaces[0]['name']}'")
    else:
        print("No network interfaces found!")
        