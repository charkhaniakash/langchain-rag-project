"""
MCP Tools Module
================

This module defines and manages MCP (Model Context Protocol) tools that
the LLM can call to perform actions like getting weather or searching Wikipedia.

Key Concepts:
- MCP tools are functions that the LLM can invoke
- Each tool has a clear schema (name, description, parameters)
- The orchestrator parses LLM output to detect tool calls
- Results are fed back to the LLM for final response generation

Available Tools:
1. get_weather(location) - Fetches weather data from Open-Meteo API
2. search_wikipedia(query) - Searches Wikipedia and returns summary
"""

import httpx
import logging
from typing import Dict, Any, List, Optional
import json
import wikipediaapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPTools:
    """
    Manages MCP tools that can be called by the LLM.
    
    This class:
    1. Defines available tools with their schemas
    2. Provides methods to execute each tool
    3. Returns tool results in a structured format
    """
    
    def __init__(self):
        """Initialize MCP tools and external API clients."""
        # Wikipedia API client
        # User agent is required by Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='VoiceAgent/1.0 (Educational Project)'
        )
        
        # HTTP client for API calls
        self.http_client = httpx.Client(timeout=10.0)
        
        logger.info("MCP Tools initialized")
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get definitions of all available tools.
        
        Returns:
            List of tool schemas that describe:
            - name: Tool identifier
            - description: What the tool does
            - parameters: Required/optional arguments
        
        This schema helps the LLM understand when and how to use each tool.
        """
        return [
            {
                "name": "get_weather",
                "description": "Get current weather information for a specific location. Returns temperature, conditions, humidity, and wind speed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or location (e.g., 'London', 'New York', 'Tokyo')"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "search_wikipedia",
                "description": "Search Wikipedia for information about a topic and return a summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The topic or query to search for on Wikipedia"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def get_weather(self, location: str) -> Dict[str, Any]:
        """
        Get weather information for a location using Open-Meteo API.
        
        Args:
            location: City name or location string
        
        Returns:
            Dictionary containing:
            - temperature: Current temperature in Celsius
            - conditions: Weather description
            - humidity: Relative humidity percentage
            - wind_speed: Wind speed in km/h
            - location: Resolved location name
        
        How it works:
        1. Geocode the location to get coordinates (lat/lon)
        2. Call Open-Meteo weather API with coordinates
        3. Parse and return weather data
        
        Open-Meteo is a free weather API that doesn't require an API key.
        """
        try:
            logger.info(f"Fetching weather for: {location}")
            
            # Step 1: Geocode location to get coordinates
            # We use Open-Meteo's geocoding API
            geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
            geocode_params = {
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json"
            }
            
            geocode_response = self.http_client.get(geocode_url, params=geocode_params)
            geocode_response.raise_for_status()
            geocode_data = geocode_response.json()
            
            if not geocode_data.get("results"):
                return {
                    "error": f"Location '{location}' not found",
                    "success": False
                }
            
            # Extract coordinates
            result = geocode_data["results"][0]
            latitude = result["latitude"]
            longitude = result["longitude"]
            location_name = result["name"]
            country = result.get("country", "")
            
            logger.info(f"Location resolved: {location_name}, {country} ({latitude}, {longitude})")
            
            # Step 2: Get weather data
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh"
            }
            
            weather_response = self.http_client.get(weather_url, params=weather_params)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            # Parse current weather
            current = weather_data["current"]
            
            # Weather code to description mapping (WMO Weather interpretation codes)
            weather_codes = {
                0: "Clear sky",
                1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                45: "Foggy", 48: "Depositing rime fog",
                51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
                61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
                71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
                80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
                95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
            }
            
            weather_code = current["weather_code"]
            conditions = weather_codes.get(weather_code, "Unknown conditions")
            
            return {
                "success": True,
                "location": f"{location_name}, {country}",
                "temperature": current["temperature_2m"],
                "conditions": conditions,
                "humidity": current["relative_humidity_2m"],
                "wind_speed": current["wind_speed_10m"],
                "unit": "Celsius"
            }
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching weather: {e}")
            return {"error": f"Failed to fetch weather data: {str(e)}", "success": False}
        except Exception as e:
            logger.error(f"Error in get_weather: {e}")
            return {"error": str(e), "success": False}
    
    def search_wikipedia(self, query: str) -> Dict[str, Any]:
        """
        Search Wikipedia for information about a query.
        
        Args:
            query: The topic or search term
        
        Returns:
            Dictionary containing:
            - title: Article title
            - summary: Brief summary (first paragraph)
            - url: Link to the full article
        
        How it works:
        1. Search Wikipedia for the query
        2. Get the top result page
        3. Extract summary (first 3 sentences)
        4. Return formatted result
        
        Uses wikipedia-api library which is free and doesn't need API keys.
        """
        try:
            logger.info(f"Searching Wikipedia for: {query}")
            
            # Get Wikipedia page
            page = self.wiki.page(query)
            
            # Check if page exists
            if not page.exists():
                # Try searching for the query
                logger.info(f"Page not found directly, trying search...")
                # Note: wikipedia-api doesn't have built-in search
                # For production, you might want to use the Wikipedia API directly
                return {
                    "error": f"No Wikipedia article found for '{query}'",
                    "success": False
                }
            
            # Extract summary (first paragraph, max 500 chars)
            summary = page.summary
            if len(summary) > 500:
                summary = summary[:497] + "..."
            
            return {
                "success": True,
                "title": page.title,
                "summary": summary,
                "url": page.fullurl
            }
            
        except Exception as e:
            logger.error(f"Error in search_wikipedia: {e}")
            return {"error": str(e), "success": False}
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Dictionary of parameters for the tool
        
        Returns:
            Result from the tool execution
        
        This is the main entry point for tool execution.
        The orchestrator calls this method when the LLM requests a tool.
        """
        logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
        
        # Route to appropriate tool
        if tool_name == "get_weather":
            return self.get_weather(parameters.get("location", ""))
        
        elif tool_name == "search_wikipedia":
            return self.search_wikipedia(parameters.get("query", ""))
        
        else:
            return {
                "error": f"Unknown tool: {tool_name}",
                "success": False
            }
    
    def format_tool_result(self, result: Dict[str, Any]) -> str:
        """
        Format a tool result into a human-readable string.
        
        This is used to present tool results to the LLM in a clear format.
        """
        if not result.get("success", False):
            return f"Error: {result.get('error', 'Unknown error')}"
        
        # Remove success flag for cleaner output
        formatted = dict(result)
        formatted.pop("success", None)
        
        return json.dumps(formatted, indent=2)


# Example usage
if __name__ == "__main__":
    # Initialize tools
    tools = MCPTools()
    
    print("Available Tools:")
    print("=" * 50)
    for tool in tools.get_tool_definitions():
        print(f"\nTool: {tool['name']}")
        print(f"Description: {tool['description']}")
        print(f"Parameters: {json.dumps(tool['parameters'], indent=2)}")
    
    print("\n" + "=" * 50)
    print("\nTesting get_weather tool:")
    weather_result = tools.execute_tool("get_weather", {"location": "London"})
    print(tools.format_tool_result(weather_result))
    
    print("\n" + "=" * 50)
    print("\nTesting search_wikipedia tool:")
    wiki_result = tools.execute_tool("search_wikipedia", {"query": "Artificial Intelligence"})
    print(tools.format_tool_result(wiki_result))