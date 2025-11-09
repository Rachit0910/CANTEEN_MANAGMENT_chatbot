from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI(
    title="Canteen Chatbot API",
    description="REST API for Canteen Chatbot - Chat, Analytics, and Business Insights",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        api_connected = True
    except Exception as e:
        print(f"Gemini API Error: {e}")
        model = None
        api_connected = False
else:
    model = None
    api_connected = False
    print("Warning: GEMINI_API_KEY not set. AI features will be limited.")

def load_menu():
    try:
        with open("menu.json", "r", encoding="utf-8") as f:
            menu_data = json.load(f)
        menu_dict = {}
        for item in menu_data:
            menu_dict[item["item_name"].lower()] = {
                "name": item["item_name"],
                "price": item["price"],
                "image_url": item.get("image_url", "")
            }
        return menu_dict, menu_data
    except Exception as e:
        print(f"Error loading menu.json: {e}")
        return {}, []

def load_data():
    csv_file = "canteen_4000_700_customers_oct2025.csv"
    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        
        column_mapping = {
            'Item': 'item',
            'Price': 'price',
            'Time': 'time',
            'Date': 'date',
            'Quantity': 'quantity',
            'Total Amount': 'total_amount',
            'Customer ID': 'customer_id',
            'Payment Method': 'payment_method',
            'Customer Satisfaction': 'satisfaction',
            'Weather': 'weather'
        }
        
        temp_cols = [col for col in df.columns if 'temperature' in col.lower() or 'temp' in col.lower()]
        if temp_cols:
            column_mapping[temp_cols[0]] = 'temperature'
        
        df = df.rename(columns=column_mapping)
        df.columns = df.columns.str.lower()
        
        if 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.day_name()
            df['time_slot'] = pd.cut(df['hour'], 
                                   bins=[0, 8, 12, 15, 18, 24],
                                   labels=['Early Morning (0-8)', 'Morning (8-12)', 
                                          'Afternoon (12-15)', 'Evening (15-18)', 'Night (18-24)'],
                                   include_lowest=True)
        
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

menu_dict, menu_data = load_menu()
df = load_data()

if df is None:
    raise RuntimeError("Failed to load dataset. Please check the CSV file.")

def get_peak_sales_time(item_name=None):
    if item_name:
        item_df = df[df['item'].str.lower() == item_name.lower()]
        if len(item_df) == 0:
            return None
        time_counts = item_df['time_slot'].value_counts()
    else:
        time_counts = df['time_slot'].value_counts()
    
    if len(time_counts) == 0:
        return None
    
    peak_time = time_counts.idxmax()
    peak_count = int(time_counts.max())
    time_distribution = {str(k): int(v) for k, v in time_counts.to_dict().items()}
    
    return {
        "peak_time": str(peak_time),
        "peak_count": peak_count,
        "time_distribution": time_distribution
    }

def get_weather_impact():
    if 'weather' not in df.columns:
        return None
    weather_sales = df.groupby('weather').agg({
        'total_amount': ['sum', 'mean', 'count']
    }).round(2)
    
    result = []
    for weather in weather_sales.index:
        result.append({
            "weather": str(weather),
            "total_revenue": float(weather_sales.loc[weather, ('total_amount', 'sum')]),
            "avg_order_value": float(weather_sales.loc[weather, ('total_amount', 'mean')]),
            "order_count": int(weather_sales.loc[weather, ('total_amount', 'count')])
        })
    return result

def get_satisfaction_analysis():
    if 'satisfaction' not in df.columns:
        return None
    satisfaction_stats = df.groupby('satisfaction').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'item': 'count'
    }).round(2)
    
    avg_satisfaction = float(df['satisfaction'].mean())
    
    result = []
    for sat_level in satisfaction_stats.index:
        result.append({
            "satisfaction_level": float(sat_level),
            "total_revenue": float(satisfaction_stats.loc[sat_level, ('total_amount', 'sum')]),
            "avg_order_value": float(satisfaction_stats.loc[sat_level, ('total_amount', 'mean')]),
            "order_count": int(satisfaction_stats.loc[sat_level, ('item', 'count')])
        })
    
    return {
        "average_satisfaction": avg_satisfaction,
        "distribution": result
    }

def get_payment_method_analysis():
    if 'payment_method' not in df.columns:
        return None
    payment_stats = df.groupby('payment_method').agg({
        'total_amount': ['sum', 'mean', 'count']
    }).round(2)
    
    result = []
    for method in payment_stats.index:
        result.append({
            "payment_method": str(method),
            "total_revenue": float(payment_stats.loc[method, ('total_amount', 'sum')]),
            "avg_order_value": float(payment_stats.loc[method, ('total_amount', 'mean')]),
            "order_count": int(payment_stats.loc[method, ('total_amount', 'count')])
        })
    return result

def get_temperature_impact():
    if 'temperature' not in df.columns:
        return None
    df_temp = df.copy()
    df_temp['temp_range'] = pd.cut(df_temp['temperature'], 
                                   bins=[0, 20, 25, 30, 35, 50],
                                   labels=['Cold (<20°C)', 'Cool (20-25°C)', 
                                          'Moderate (25-30°C)', 'Warm (30-35°C)', 'Hot (>35°C)'])
    temp_sales = df_temp.groupby('temp_range').agg({
        'total_amount': ['sum', 'mean', 'count']
    }).round(2)
    
    result = []
    for temp_range in temp_sales.index:
        result.append({
            "temperature_range": str(temp_range),
            "total_revenue": float(temp_sales.loc[temp_range, ('total_amount', 'sum')]),
            "avg_order_value": float(temp_sales.loc[temp_range, ('total_amount', 'mean')]),
            "order_count": int(temp_sales.loc[temp_range, ('total_amount', 'count')])
        })
    return result

def get_customer_behavior():
    customer_stats = df.groupby('customer_id').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'item': 'nunique',
        'satisfaction': 'mean'
    }).round(2)
    customer_stats.columns = ['total_spent', 'avg_order_value', 'visit_count', 'unique_items', 'avg_satisfaction']
    top_customers = customer_stats.nlargest(10, 'total_spent')
    
    result = []
    for customer_id, data in top_customers.iterrows():
        result.append({
            "customer_id": str(customer_id),
            "total_spent": float(data['total_spent']),
            "avg_order_value": float(data['avg_order_value']),
            "visit_count": int(data['visit_count']),
            "unique_items": int(data['unique_items']),
            "avg_satisfaction": float(data['avg_satisfaction']) if not pd.isna(data['avg_satisfaction']) else None
        })
    
    return result

def get_revenue_trends():
    if 'datetime' not in df.columns:
        return None
    df_time = df.copy()
    df_time['date_only'] = df_time['datetime'].dt.date
    daily_revenue = df_time.groupby('date_only')['total_amount'].sum().reset_index()
    daily_revenue.columns = ['date', 'revenue']
    
    result = []
    for _, row in daily_revenue.iterrows():
        result.append({
            "date": str(row['date']),
            "revenue": float(row['revenue'])
        })
    return result

def get_comprehensive_insights():
    peak_info = get_peak_sales_time()
    top_items_dict = df['item'].value_counts().head(5).to_dict()
    top_items = {str(k): int(v) for k, v in top_items_dict.items()}
    
    insights = {
        'total_revenue': float(df['total_amount'].sum()),
        'avg_order_value': float(df['total_amount'].mean()),
        'total_orders': int(len(df)),
        'unique_customers': int(df['customer_id'].nunique()),
        'unique_items': int(df['item'].nunique()),
        'top_items': top_items,
        'peak_time': str(peak_info['peak_time']) if peak_info and peak_info.get('peak_time') else None,
        'avg_satisfaction': float(df['satisfaction'].mean()) if 'satisfaction' in df.columns else None,
    }
    return insights

def get_most_selling_items(n=10):
    item_sales = df['item'].value_counts().head(n)
    result = []
    for item, count in item_sales.items():
        price = get_item_price(item) or 0
        revenue = float(df[df['item'] == item]['total_amount'].sum())
        result.append({
            "item": item,
            "order_count": int(count),
            "price": price,
            "revenue": revenue
        })
    return result

def get_least_selling_items(n=10):
    item_sales = df['item'].value_counts().tail(n)
    result = []
    for item, count in item_sales.items():
        price = get_item_price(item) or 0
        revenue = float(df[df['item'] == item]['total_amount'].sum())
        result.append({
            "item": item,
            "order_count": int(count),
            "price": price,
            "revenue": revenue
        })
    return result

def get_item_price(item_name):
    item_lower = item_name.lower()
    if item_lower in menu_dict:
        return menu_dict[item_lower]["price"]
    return None

def get_item_from_menu(item_name):
    item_lower = item_name.lower()
    if item_lower in menu_dict:
        return menu_dict[item_lower]
    return None

def get_gemini_response(prompt, context="", system_prompt=None):
    if not model or not api_connected:
        return None
    
    try:
        if system_prompt:
            full_context = f"{system_prompt}\n\n{context}\n\n{prompt}"
        else:
            full_context = f"{context}\n\n{prompt}" if context else prompt
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        response = model.generate_content(
            full_context,
            generation_config=generation_config
        )
        
        if hasattr(response, 'text'):
            return response.text
        else:
            return None
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

def build_comprehensive_context(user_query):
    context = "CANTEEN BUSINESS CONTEXT:\n\n"
    
    insights = get_comprehensive_insights()
    context += f"BUSINESS OVERVIEW:\n"
    context += f"- Total Revenue: ₹{insights['total_revenue']:,.2f}\n"
    context += f"- Total Orders: {insights['total_orders']:,}\n"
    context += f"- Unique Customers: {insights['unique_customers']:,}\n"
    context += f"- Menu Items: {insights['unique_items']}\n"
    context += f"- Average Order Value: ₹{insights['avg_order_value']:,.2f}\n"
    if insights['avg_satisfaction']:
        context += f"- Average Customer Satisfaction: {insights['avg_satisfaction']:.2f}/5\n"
    context += f"- Peak Sales Time: {insights['peak_time']}\n\n"
    
    context += f"TOP SELLING ITEMS:\n"
    for item, count in list(insights['top_items'].items())[:5]:
        price = get_item_price(item) or "N/A"
        revenue = df[df['item'] == item]['total_amount'].sum()
        context += f"- {item}: {count} orders, Price: ₹{price}, Revenue: ₹{revenue:,.2f}\n"
    context += "\n"
    
    return context

class ChatRequest(BaseModel):
    query: str
    include_context: bool = True

class ChatResponse(BaseModel):
    response: str
    ai_response: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class ItemRequest(BaseModel):
    item_name: str

@app.get("/")
async def root():
    return {
        "message": "Canteen Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat",
            "menu": "/api/menu",
            "analytics": "/api/analytics",
            "graphs": "/api/graphs",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "data_loaded": df is not None,
        "menu_loaded": len(menu_dict) > 0,
        "ai_connected": api_connected
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        user_input = request.query
        user_input_lower = user_input.lower()
        
        context = build_comprehensive_context(user_input) if request.include_context else ""
        
        system_prompt = """You are a professional, intelligent canteen assistant powered by advanced AI. 
Your role is to:
1. Help customers with menu inquiries, prices, descriptions, and recommendations
2. Provide management with data-driven insights, analytics, and business recommendations
3. Answer any questions about the canteen business intelligently
4. Provide context-aware, personalized recommendations
5. Be friendly, professional, and helpful

Always use the provided data and context to give accurate, data-driven answers."""
        
        ai_response = None
        if api_connected:
            prompt = f"User query: '{user_input}'. Respond naturally and helpfully using the business context."
            ai_response = get_gemini_response(prompt, context, system_prompt)
        
        response_text = ""
        data = {}
        
        if any(word in user_input_lower for word in ['price', 'cost', 'how much', '₹', 'rupee']):
            found_item = None
            for item_name in menu_dict.keys():
                if item_name in user_input_lower:
                    found_item = item_name
                    break
            
            if found_item:
                item_info = get_item_from_menu(found_item)
                if item_info:
                    response_text = f"Price of {item_info['name']}: ₹{item_info['price']}"
                    data = {
                        "item": item_info['name'],
                        "price": item_info['price'],
                        "image_url": item_info.get('image_url', '')
                    }
                    peak_info = get_peak_sales_time(item_info['name'])
                    if peak_info:
                        data['peak_sales_time'] = peak_info['peak_time']
                        data['sales_count'] = peak_info['peak_count']
        
        elif any(word in user_input_lower for word in ['analytics', 'sales', 'revenue', 'statistics', 'stats']):
            insights = get_comprehensive_insights()
            response_text = f"Business Analytics:\nTotal Revenue: ₹{insights['total_revenue']:,.2f}\nTotal Orders: {insights['total_orders']:,}\nAverage Order Value: ₹{insights['avg_order_value']:,.2f}"
            data = insights
        
        elif any(word in user_input_lower for word in ['top', 'most', 'popular', 'best selling']):
            top_items = get_most_selling_items(10)
            response_text = "Top Selling Items:"
            data = {"top_items": top_items}
        
        if not response_text and ai_response:
            response_text = ai_response
        elif not response_text:
            response_text = "I can help you with menu items, prices, analytics, and recommendations. Please ask me anything about the canteen!"
        
        return ChatResponse(
            response=response_text,
            ai_response=ai_response,
            data=data if data else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/api/menu")
async def get_menu():
    return {
        "menu": menu_data,
        "total_items": len(menu_data)
    }

@app.get("/api/menu/{item_name}")
async def get_menu_item(item_name: str):
    item_info = get_item_from_menu(item_name)
    if not item_info:
        raise HTTPException(status_code=404, detail="Item not found")
    
    peak_info = get_peak_sales_time(item_name)
    item_sales = df[df['item'].str.lower() == item_name.lower()]
    total_revenue = float(item_sales['total_amount'].sum()) if len(item_sales) > 0 else 0
    order_count = len(item_sales)
    
    return {
        "item": item_info['name'],
        "price": item_info['price'],
        "image_url": item_info.get('image_url', ''),
        "peak_sales_time": peak_info['peak_time'] if peak_info else None,
        "total_revenue": total_revenue,
        "order_count": order_count
    }

@app.get("/api/menu/search/{query}")
async def search_menu(query: str):
    query_lower = query.lower()
    results = []
    for item in menu_data:
        if query_lower in item['item_name'].lower():
            results.append(item)
    return {
        "query": query,
        "results": results,
        "count": len(results)
    }

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    return get_comprehensive_insights()

@app.get("/api/analytics/top-items")
async def get_top_items(limit: int = Query(10, ge=1, le=50)):
    return {
        "top_items": get_most_selling_items(limit)
    }

@app.get("/api/analytics/least-items")
async def get_least_items(limit: int = Query(10, ge=1, le=50)):
    return {
        "least_items": get_least_selling_items(limit)
    }

@app.get("/api/analytics/weather-impact")
async def get_weather_impact_data():
    result = get_weather_impact()
    if result is None:
        raise HTTPException(status_code=404, detail="Weather data not available")
    return {"weather_impact": result}

@app.get("/api/analytics/satisfaction")
async def get_satisfaction_data():
    result = get_satisfaction_analysis()
    if result is None:
        raise HTTPException(status_code=404, detail="Satisfaction data not available")
    return result

@app.get("/api/analytics/payment-methods")
async def get_payment_methods_data():
    result = get_payment_method_analysis()
    if result is None:
        raise HTTPException(status_code=404, detail="Payment method data not available")
    return {"payment_methods": result}

@app.get("/api/analytics/temperature-impact")
async def get_temperature_impact_data():
    result = get_temperature_impact()
    if result is None:
        raise HTTPException(status_code=404, detail="Temperature data not available")
    return {"temperature_impact": result}

@app.get("/api/analytics/customer-behavior")
async def get_customer_behavior_data(limit: int = Query(10, ge=1, le=50)):
    customers = get_customer_behavior()
    return {
        "top_customers": customers[:limit]
    }

@app.get("/api/analytics/revenue-trends")
async def get_revenue_trends_data():
    result = get_revenue_trends()
    if result is None:
        raise HTTPException(status_code=404, detail="Revenue trends data not available")
    return {"revenue_trends": result}

@app.get("/api/analytics/peak-time")
async def get_peak_time(item_name: Optional[str] = None):
    result = get_peak_sales_time(item_name)
    if result is None:
        raise HTTPException(status_code=404, detail="Peak time data not available")
    return result

@app.get("/api/graphs/sales-by-time")
async def get_sales_by_time_graph(item_name: Optional[str] = None):
    if item_name:
        item_df = df[df['item'].str.lower() == item_name.lower()]
        if len(item_df) == 0:
            raise HTTPException(status_code=404, detail="Item not found")
        time_counts = item_df['time_slot'].value_counts()
    else:
        time_counts = df['time_slot'].value_counts()
    
    chart_data = {
        "labels": [str(x) for x in time_counts.index],
        "values": [int(x) for x in time_counts.values],
        "data": [{"time_slot": str(x), "count": int(y)} for x, y in zip(time_counts.index, time_counts.values)]
    }
    return chart_data

@app.get("/api/graphs/revenue-by-item")
async def get_revenue_by_item_graph(limit: int = Query(10, ge=1, le=50)):
    item_revenue = df.groupby('item')['total_amount'].sum().sort_values(ascending=False).head(limit)
    
    chart_data = {
        "labels": [str(x) for x in item_revenue.index],
        "values": [float(x) for x in item_revenue.values],
        "data": [{"item": str(x), "revenue": float(y)} for x, y in zip(item_revenue.index, item_revenue.values)]
    }
    return chart_data

@app.get("/api/graphs/weather-impact")
async def get_weather_impact_graph():
    result = get_weather_impact()
    if result is None:
        raise HTTPException(status_code=404, detail="Weather data not available")
    
    chart_data = {
        "labels": [x["weather"] for x in result],
        "revenue": [x["total_revenue"] for x in result],
        "orders": [x["order_count"] for x in result],
        "data": result
    }
    return chart_data

@app.get("/api/graphs/satisfaction-distribution")
async def get_satisfaction_distribution_graph():
    result = get_satisfaction_analysis()
    if result is None:
        raise HTTPException(status_code=404, detail="Satisfaction data not available")
    
    chart_data = {
        "labels": [f"{x['satisfaction_level']}/5" for x in result["distribution"]],
        "orders": [x["order_count"] for x in result["distribution"]],
        "revenue": [x["total_revenue"] for x in result["distribution"]],
        "average_satisfaction": result["average_satisfaction"],
        "data": result["distribution"]
    }
    return chart_data

@app.get("/api/graphs/payment-methods")
async def get_payment_methods_graph():
    result = get_payment_method_analysis()
    if result is None:
        raise HTTPException(status_code=404, detail="Payment method data not available")
    
    chart_data = {
        "labels": [x["payment_method"] for x in result],
        "revenue": [x["total_revenue"] for x in result],
        "orders": [x["order_count"] for x in result],
        "data": result
    }
    return chart_data

@app.get("/api/graphs/revenue-trends")
async def get_revenue_trends_graph():
    result = get_revenue_trends()
    if result is None:
        raise HTTPException(status_code=404, detail="Revenue trends data not available")
    
    chart_data = {
        "labels": [x["date"] for x in result],
        "values": [x["revenue"] for x in result],
        "data": result
    }
    return chart_data

@app.get("/api/graphs/top-customers")
async def get_top_customers_graph(limit: int = Query(10, ge=1, le=50)):
    customers = get_customer_behavior()
    
    chart_data = {
        "labels": [f"Customer {i+1}" for i in range(min(limit, len(customers)))],
        "revenue": [x["total_spent"] for x in customers[:limit]],
        "visits": [x["visit_count"] for x in customers[:limit]],
        "data": customers[:limit]
    }
    return chart_data

@app.get("/api/recommendations")
async def get_recommendations(
    weather: Optional[str] = None,
    time_of_day: Optional[str] = None,
    temperature: Optional[float] = None
):
    context_items = []
    
    if weather and 'weather' in df.columns:
        weather_items = df[df['weather'] == weather]['item'].value_counts().head(5)
        context_items.extend(weather_items.index.tolist())
    
    if time_of_day and 'time_slot' in df.columns:
        time_items = df[df['time_slot'] == time_of_day]['item'].value_counts().head(5)
        context_items.extend(time_items.index.tolist())
    
    recommendations = list(set(context_items))[:10]
    
    result = []
    for item in recommendations:
        item_info = get_item_from_menu(item)
        if item_info:
            result.append({
                "item": item_info['name'],
                "price": item_info['price'],
                "image_url": item_info.get('image_url', '')
            })
    
    return {
        "recommendations": result,
        "context": {
            "weather": weather,
            "time_of_day": time_of_day,
            "temperature": temperature
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
