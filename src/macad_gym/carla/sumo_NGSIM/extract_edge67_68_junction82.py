#!/usr/bin/env python3
"""
提取edge67、edge68和junction82坐标信息
"""

import sumolib

def main():
    # 加载网络文件
    net = sumolib.net.readNet("carla_town04_official.net.xml")
    
    # print("=== Edge0坐标信息 ===")
    # edge0 = net.getEdge("0")
    # print(f"长度: {edge0.getLength():.1f}米, 车道数: {len(edge0.getLanes())}")
    # for lane in edge0.getLanes():
    #     shape = lane.getShape()
    #     start = shape[0]
    #     end = shape[-1]
    #     print(f"车道{lane.getID()}: 起点({start[0]:.2f}, {start[1]:.2f}) 终点({end[0]:.2f}, {end[1]:.2f})")
    #
    # print("\n=== Edge1坐标信息 ===")
    # edge1 = net.getEdge("1")
    # print(f"长度: {edge1.getLength():.1f}米, 车道数: {len(edge1.getLanes())}")
    # for lane in edge1.getLanes():
    #     shape = lane.getShape()
    #     start = shape[0]
    #     end = shape[-1]
    #     print(f"车道{lane.getID()}: 起点({start[0]:.2f}, {start[1]:.2f}) 终点({end[0]:.2f}, {end[1]:.2f})")
    #
    #
    # print("\n=== Edge2坐标信息 ===")
    # edge2 = net.getEdge("2")
    # print(f"长度: {edge2.getLength():.1f}米, 车道数: {len(edge2.getLanes())}")
    # for lane in edge2.getLanes():
    #     shape = lane.getShape()
    #     start = shape[0]
    #     end = shape[-1]
    #     print(f"车道{lane.getID()}: 起点({start[0]:.2f}, {start[1]:.2f}) 终点({end[0]:.2f}, {end[1]:.2f})")
    print("\n=== Edge40坐标信息 ===")
    edge40 = net.getEdge("40")
    print(f"长度: {edge40.getLength():.1f}米, 车道数: {len(edge40.getLanes())}")
    for lane in edge40.getLanes():
        shape = lane.getShape()
        start = shape[0]
        end = shape[-1]
        print(f"车道{lane.getID()}: 起点({start[0]:.2f}, {start[1]:.2f}) 终点({end[0]:.2f}, {end[1]:.2f})")

    print("\n=== Edge39坐标信息 ===")
    edge39 = net.getEdge("39")
    print(f"长度: {edge39.getLength():.1f}米, 车道数: {len(edge39.getLanes())}")
    for lane in edge39.getLanes():
        shape = lane.getShape()
        start = shape[0]
        end = shape[-1]
        print(f"车道{lane.getID()}: 起点({start[0]:.2f}, {start[1]:.2f}) 终点({end[0]:.2f}, {end[1]:.2f})")

    # print("\n=== Edge4坐标信息 ===")
    # edge4 = net.getEdge("4")
    # print(f"长度: {edge4.getLength():.1f}米, 车道数: {len(edge4.getLanes())}")
    # for lane in edge4.getLanes():
    #     shape = lane.getShape()
    #     start = shape[0]
    #     end = shape[-1]
    #     print(f"车道{lane.getID()}: 起点({start[0]:.2f}, {start[1]:.2f}) 终点({end[0]:.2f}, {end[1]:.2f})")
    
    # print("\n=== Junction82坐标信息 ===")
    # junction82 = net.getNode("82")
    # coord = junction82.getCoord()
    # print(f"路口位置: ({coord[0]:.2f}, {coord[1]:.2f})")
    # print(f"连接边: {[e.getID() for e in junction82.getIncoming() + junction82.getOutgoing()]}")

if __name__ == "__main__":
    main()
