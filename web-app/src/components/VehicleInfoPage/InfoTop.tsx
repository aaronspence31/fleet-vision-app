"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";

const InfoTop = () => {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current) return;
    const el = d3.select(ref.current);
    el.selectAll("*").remove();
    const svg = el.append("svg").attr("width", "100%").attr("height", "100%");
    const g = svg.append("g");

    // Bottom line
    g.append("image")
      .attr("href", "/images/up_bottom_line.svg")
      .attr("x", "0")
      .attr("y", "30px")
      .attr("width", "740px")
      .attr("height", "50px");

    // Progress line on top
    g.append("image")
      .attr("href", "/images/up_top_dots.svg")
      .attr("x", "30px")
      .attr("y", "5px")
      .attr("width", "680px")
      .attr("height", "30px");

    // Left arrow
    g.append("image")
      .attr("href", "/images/left-arrow.svg")
      .attr("x", "80px")
      .attr("y", "35px")
      .attr("width", "30px")
      .attr("height", "30px");

    // Right arrow
    g.append("image")
      .attr("href", "/images/right-arrow.svg")
      .attr("x", "630px")
      .attr("y", "35px")
      .attr("width", "30px")
      .attr("height", "30px");

    // Gasoline pump
    g.append("image")
      .attr("href", "/images/gasoline-pump.svg")
      .attr("x", "130px")
      .attr("y", "37px")
      .attr("width", "25px")
      .attr("height", "25px");

    // Kilometers remaining text
    g.append("text")
      .text("130 km")
      .attr("x", "195px")
      .attr("y", "55px")
      .attr("font-size", "16")
      .attr("text-anchor", "middle")
      .attr("fill", "#000000");

    // Red line
    g.append("image")
      .attr("href", "/images/red-line.svg")
      .attr("x", "235px")
      .attr("y", "35px")
      .attr("width", "25px")
      .attr("height", "25px");

    // Circles LTE signal
    let x = 280;
    d3.range(5).forEach(() => {
      g.append("circle")
        .attr("cx", x)
        .attr("cy", "48px")
        .attr("r", "3")
        .attr("fill", "#000000");
      x += 10;
    });

    // LTE signal text
    g.append("text")
      .text("LTE")
      .attr("x", "347px")
      .attr("y", "53px")
      .attr("font-size", "14")
      .attr("text-anchor", "middle")
      .attr("fill", "#000000");

    // Location image
    g.append("image")
      .attr("href", "/images/location.svg")
      .attr("x", "380px")
      .attr("y", "42px")
      .attr("width", "15px")
      .attr("height", "15px");

    // Bluetooth image
    g.append("image")
      .attr("href", "/images/bluetooth.svg")
      .attr("x", "400px")
      .attr("y", "40px")
      .attr("width", "18px")
      .attr("height", "18px");

    // Battery text
    g.append("text")
      .text("46%")
      .attr("x", "450px")
      .attr("y", "54px")
      .attr("font-size", "14")
      .attr("text-anchor", "middle")
      .attr("fill", "#000000");

    // Battery image
    g.append("image")
      .attr("href", "/images/battery.svg")
      .attr("x", "470px")
      .attr("y", "34px")
      .attr("width", "30px")
      .attr("height", "30px");

    // Red line
    g.append("image")
      .attr("href", "/images/red-line.svg")
      .attr("x", "520px")
      .attr("y", "35px")
      .attr("width", "25px")
      .attr("height", "25px");

    // Temperature text
    g.append("text")
      .text("-12 °C")
      .attr("x", "580px")
      .attr("y", "54px")
      .attr("font-size", "14")
      .attr("text-anchor", "middle")
      .attr("fill", "#000000");
  }, []);

  return <div style={{ width: 1000 }} className="info-top" ref={ref}></div>;
};

export default InfoTop;
