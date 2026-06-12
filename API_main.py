import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import dashscope
from dashscope import Generation
import httpx

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==================== 初始化配置 ====================
# 尝试从多个位置加载环境变量
env_paths = [
    '.env',
    './.env',
    os.path.join(os.path.dirname(__file__), '.env')
]

loaded = False
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        loaded = True
        logger.info(f"✅ 从 {env_path} 加载环境变量")
        break

if not loaded:
    logger.warning("⚠️ 未找到 .env 文件，将使用系统环境变量")

# API Keys 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or "sk-51c6838969dc421f847302adadc60257"
AMAP_KEY = os.getenv("AMAP_KEY") or "a5a982b43d4cb9ae9031358d2410d047"

# 验证必需配置
if not DASHSCOPE_API_KEY:
    logger.warning("DASHSCOPE_API_KEY 未配置，AI功能可能受限")
else:
    logger.info("✅ DASHSCOPE_API_KEY 已配置")
    
if not AMAP_KEY:
    logger.warning("AMAP_KEY 未配置，真实数据获取功能可能受限")
else:
    logger.info("✅ AMAP_KEY 已配置")

dashscope.api_key = DASHSCOPE_API_KEY

app = FastAPI(title="AI旅行规划API", version="2.1.0")

# 配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 常量定义 ====================
# 高德POI分类编码
POI_TYPES = {
    "景点": "110000",
    "美食": "050000",
    "酒店": "200000",
    "火车站": "150500",
    "机场": "150400",
}

# 预算等级对应价格范围（占总预算的40-50%用于住宿）
BUDGET_PRICE_MAP = {
    "经济": "100-200元/晚/人",
    "舒适": "200-350元/晚/人",
    "豪华": "400-700元/晚/人",
}

# 预算分配比例（占总预算百分比）
BUDGET_ALLOCATION = {
    "住宿": 40,    # 酒店住宿
    "餐饮": 30,    # 三餐及小吃
    "交通": 15,    # 市内交通、打车等
    "门票": 10,    # 景点门票
    "其他": 5,     # 购物、应急等
}

def calculate_budget_allocation(total_budget: int, days: int, people_count: int) -> dict:
    """
    根据总预算、天数和人数计算合理的预算分配
    
    :param total_budget: 总预算金额（元）
    :param days: 旅行天数
    :param people_count: 旅行人数
    :return: 各项目的预算分配字典
    """
    # 人均总预算
    per_person_total = total_budget / people_count
    
    # 人均每日预算
    per_person_daily = per_person_total / days
    
    # 各项目预算分配
    allocation = {
        "人均每日预算": round(per_person_daily, 2),
        "人均住宿预算（每晚）": round(per_person_daily * BUDGET_ALLOCATION["住宿"] / 100, 2),
        "人均餐饮预算（每天）": round(per_person_daily * BUDGET_ALLOCATION["餐饮"] / 100, 2),
        "人均交通预算（每天）": round(per_person_daily * BUDGET_ALLOCATION["交通"] / 100, 2),
        "人均门票预算（每天）": round(per_person_daily * BUDGET_ALLOCATION["门票"] / 100, 2),
        "人均其他预算（每天）": round(per_person_daily * BUDGET_ALLOCATION["其他"] / 100, 2),
        "总住宿预算": round(total_budget * BUDGET_ALLOCATION["住宿"] / 100, 2),
        "每晚住宿预算（人均）": round((total_budget * BUDGET_ALLOCATION["住宿"] / 100) / days / people_count, 2),
    }
    
    return allocation

# ==================== 数据模型定义 ====================
class PlanRequest(BaseModel):
    """前端请求参数校验模型"""
    destination: str = Field(default="漳州市", description="目的地城市")
    days: int = Field(default=5, ge=1, le=30, description="旅行天数")
    budget: str = Field(default="舒适", description="预算等级")
    total_budget: Optional[int] = Field(None, ge=500, le=100000, description="总预算金额（元）")
    start_date: Optional[str] = Field(None, description="出发日期")
    preferences: List[str] = Field(default_factory=list, description="旅行偏好")
    people_count: int = Field(default=2, ge=1, le=50, description="旅行人数")

class PlanResponse(BaseModel):
    """统一响应格式"""
    status: str = "success"
    message: str = "success"
    data: Optional[Dict[str, Any]] = None

# ==================== 工具函数 ====================
async def get_city_code(city: str) -> Optional[str]:
    """获取城市编码（地理编码）"""
    if not AMAP_KEY:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://restapi.amap.com/v3/geocode/geo",
                params={"address": city, "key": AMAP_KEY}
            )
            response.raise_for_status()
            data = response.json()
            geocodes = data.get("geocodes", [])
            return geocodes[0]["adcode"] if geocodes else None
    except Exception as e:
        logger.error(f"获取城市编码失败 - {city}: {str(e)}")
        return None

async def fetch_poi_from_amap(city_code: str, keywords: str, poi_type: str, limit: int = 5) -> List[Dict[str, Any]]:
    """通用POI搜索函数"""
    if not AMAP_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://restapi.amap.com/v3/place/text",
                params={
                    "key": AMAP_KEY,
                    "city": city_code,
                    "keywords": keywords,
                    "types": poi_type,
                    "offset": str(limit),
                    "page": "1"
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("pois", [])[:limit]
    except Exception as e:
        logger.error(f"POI搜索失败 - {keywords}: {str(e)}")
        return []

# ==================== 核心业务逻辑 ====================
SYSTEM_PROMPT_EXTENDED = """
你是"小游"，一位深耕国内旅游多年的资深旅行规划专家。

【身份定位】
- 专业旅行顾问：精通国内热门旅游目的地，熟悉各城市特色景点、美食和住宿
- 精准规划师：能够根据用户需求制定合理、可行的旅行计划

【硬性约束规则 - 必须严格遵守】
=== 天数约束 ===
1. itinerary数组长度必须严格等于用户指定的旅行天数
2. 每个行程日的day字段必须从1开始连续递增（1, 2, 3...）
3. 不得遗漏任何一天，也不得多生成天数

=== 地点约束 ===
1. 必须使用用户提供的真实数据中的景点、餐厅、酒店名称
2. 严禁编造不存在的地点名称
3. 每个景点每天只能安排一次，不得重复游览同一景点
4. 推荐的地点必须位于目标城市及周边30公里合理范围内
5. 【酒店一致性约束】酒店名称必须与地址对应：
   - hotel字段必须包含酒店名称和对应的真实地址
   - 地址必须是目标城市的真实地址（如"成都市锦江区春熙路XX号"）
   - 所有行程活动中提到的"酒店"必须与hotel字段中的酒店名称完全一致
6. 【酒店位置约束】酒店地址必须根据行程合理规划：
   - 酒店应位于主要景点集中的区域或交通便利的地段（如地铁站附近、市中心商圈）
   - 酒店位置应方便前往第一天行程的第一个景点（步行或短距离车程）
   - 酒店地址应在所有景点的地理中心附近，减少整体交通时间
   - 优先选择位于热门旅游区域或景点密集区域的酒店
7. 【酒店一致性约束】整个行程必须使用同一个酒店：
   - 所有天数的hotel字段必须完全相同（酒店名称和地址都要一致）
   - 禁止每天更换不同的酒店
   - 必须从【可用酒店列表】中选择真实酒店名称和地址
8. 【酒店真实性约束】必须使用真实酒店数据：
   - hotel字段必须完全匹配【可用酒店列表】中的某一个酒店（名称和地址都要匹配）
   - 严禁编造任何酒店名称或地址
   - 如果【可用酒店列表】为空，返回空的hotel字段
   - 酒店地址必须包含具体的街道和门牌号（如"思明区漳州路36号"）
9. 【景点类型约束】以下内容严禁作为景点推荐：
   - 租借单车、电动车、自行车（如哈啰单车、共享单车）
   - 购物行为（如买特产、逛商场）
   - 交通换乘点（如地铁站、公交站、停车场）
   - 便利店、药店、银行等生活服务设施
   - 加油站、充电站
   这些内容只能在tip字段中作为交通建议提及，不能作为行程中的景点活动

=== 景点定义 ===
景点必须是具有游览价值的地点，包括：
- 自然景观：公园、山川、湖泊、海滩、森林等
- 人文景观：博物馆、古迹、寺庙、教堂、历史建筑等
- 休闲娱乐：游乐场、动物园、植物园、主题公园等
- 特色街区：古镇、老街、步行街、夜市等
- 文化场所：美术馆、图书馆、展览馆等

=== 时间约束 ===
1. 时间顺序必须合理：上午活动（08:00-12:00）、下午活动（14:00-18:00）、晚上活动（18:00-22:00）
2. 相邻活动间必须预留至少30分钟交通时间
3. 每个活动必须包含具体时间点（如08:30、14:00）
4. 禁止出现不合理时间安排（如上午9点活动直接连接晚上活动而无午餐时间）

=== 路线规划约束 ===
1. 【起点约束】每天行程必须从酒店出发开始，明确标注"从酒店出发"
2. 【路线顺序】景点安排必须考虑地理位置合理性：
   - 相邻景点距离不宜过远（建议同一区域或相邻区域）
   - 避免来回折返，按照地理位置顺序安排
   - 优先安排距离酒店较近的景点作为第一站
3. 【路线指引】每个景点活动必须包含清晰的导航信息：
   - morning格式："{时间} 从酒店出发 → {时间} 到达景点A（地址）游玩"
   - afternoon格式："{时间} 从景点A出发 → {时间} 到达景点B（地址）游玩"
   - evening格式："{时间} 从景点B出发 → {时间} 到餐厅用餐 → {时间} 返回酒店"
4. 【终点约束】每天行程最后必须返回酒店休息

=== 预算约束 ===
1. 严格按预算等级规划：经济档酒店100-200元/晚/人，舒适档200-350元/晚/人，豪华档400-700元/晚/人
2. 预算分配原则：住宿占40%、餐饮占30%、交通占15%、门票占10%、其他占5%
3. 确保餐饮预算充足，每天三餐都要有足够预算
4. 严禁出现预算严重偏差（如5000元预算规划出5万元行程）

=== 天气适配约束 ===
1. 根据实时天气调整行程安排（高温避开正午、雨天准备雨具等）
2. 在tip字段中加入针对性的天气注意事项和穿衣建议

【输出格式要求 - 严格遵守】
=== 必须返回纯JSON格式 ===
- 禁止包含任何Markdown标记（如```json）
- 禁止包含解释性文字
- 必须返回有效的JSON字符串

=== JSON结构定义 ===
{
  "itinerary": [
    {
      "day": 数字（1开始的连续整数）,
      "title": "第N天 · 主题名称"（必须包含"第N天"或"Day N"）,
      "morning": "时间点 活动描述（使用真实数据中的地点名称，包含具体时间）",
      "afternoon": "时间点 活动描述（使用真实数据中的地点名称，包含具体时间）",
      "evening": "时间点 活动描述（使用真实数据中的地点名称，包含具体时间）",
      "food": "今日美食推荐（使用真实餐厅名+推荐菜品+价格区间）",
      "hotel": "酒店名称及地址（使用真实酒店名，标注价格）",
      "tip": "今日小贴士（包含天气提醒、避坑建议等实用信息）"
    }
  ]
}

=== 字段值要求 ===
1. day：必须为从1开始的连续整数，不能跳跃或重复
2. title：必须包含"第N天"或"Day N"标识
3. morning/afternoon/evening：必须包含具体时间点（如08:30）和活动描述
4. food：必须使用真实美食数据中的餐厅名称，标注价格区间
5. hotel：必须使用真实酒店数据中的酒店名称，标注价格，符合预算要求
6. tip：必须包含实用信息，如天气提醒、交通提示、游玩建议等

【失败惩罚】
- 如果违反上述任何约束规则，将被视为输出失败
- 输出格式错误（非有效JSON）将被视为输出失败
- 编造地点名称将被视为输出失败
"""

async def get_ai_response_async(messages: List[Dict[str, str]]) -> str:
    """异步调用通义千问API"""
    if not DASHSCOPE_API_KEY:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY 未配置")
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: Generation.call(
                model="qwen-plus",
                messages=messages,
                result_format="message"
            )
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"AI调用失败: {response.message}"
            )
    except Exception as e:
        logger.error(f"AI调用异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_real_weather(city: str) -> str:
    """调用高德天气API获取实时天气"""
    if not AMAP_KEY:
        return "⚠️ 未配置高德天气API Key"
    
    try:
        city_code = await get_city_code(city)
        if not city_code:
            return f"🌤️ {city} 天气数据暂不可用"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://restapi.amap.com/v3/weather/weatherInfo",
                params={"city": city_code, "key": AMAP_KEY}
            )
            response.raise_for_status()
            data = response.json()
            lives = data.get("lives", [])
            
            if not lives:
                return f"[Weather] {city} 天气数据不可用"

            d = lives[0]
            return (f"{d['weather']} {d['temperature']}°C "
                    f"Wind:{d['winddirection']} {d['windpower']}级")
    except Exception as e:
        logger.error(f"获取天气失败 - {city}: {str(e)}")
        return "[Weather] 天气数据获取失败"

# ==================== 真实数据获取函数 ====================
async def fetch_real_spots_from_amap(city: str) -> List[Dict[str, Any]]:
    """获取真实景点数据"""
    city_code = await get_city_code(city)
    if not city_code:
        return []
    
    pois = await fetch_poi_from_amap(city_code, "景点", POI_TYPES["景点"], 5)
    return [{
        "name": poi.get("name", ""),
        "address": poi.get("address", ""),
        "location": poi.get("location", ""),
        "tel": poi.get("tel", ""),
        "type": poi.get("type", ""),
        "distance": poi.get("distance", "")
    } for poi in pois]

async def fetch_location_from_address(address: str) -> Optional[Dict[str, float]]:
    """根据地址获取经纬度坐标"""
    if not AMAP_KEY:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://restapi.amap.com/v3/geocode/geo",
                params={"address": address, "key": AMAP_KEY}
            )
            response.raise_for_status()
            data = response.json()
            geocodes = data.get("geocodes", [])
            if geocodes:
                loc = geocodes[0].get("location", "")
                if loc:
                    lng, lat = loc.split(",")
                    return {"lng": float(lng), "lat": float(lat)}
            return None
    except Exception as e:
        logger.error(f"地址转坐标失败 - {address}: {str(e)}")
        return None

async def fetch_real_food_from_amap(city: str) -> List[Dict[str, Any]]:
    """获取真实美食数据"""
    city_code = await get_city_code(city)
    if not city_code:
        return []
    
    pois = await fetch_poi_from_amap(city_code, "美食餐厅", POI_TYPES["美食"], 5)
    return [{
        "name": poi.get("name", ""),
        "address": poi.get("address", ""),
        "type": poi.get("type", ""),
        "tel": poi.get("tel", ""),
        "price_range": "人均50-100元"
    } for poi in pois]

async def fetch_real_hotel_from_amap(city: str, budget: str = "舒适", limit: int = 5) -> List[Dict[str, Any]]:
    """获取真实酒店数据"""
    city_code = await get_city_code(city)
    if not city_code:
        return []
    
    # 根据预算设置搜索关键词
    budget_keywords = {
        "经济": ["经济型酒店", "快捷酒店", "连锁酒店"],
        "舒适": ["酒店", "商务酒店", "精品酒店"],
        "豪华": ["五星级酒店", "豪华酒店", "高端酒店"]
    }
    
    keywords_list = budget_keywords.get(budget, budget_keywords["舒适"])
    all_hotels = []
    
    # 尝试多个关键词搜索
    for keyword in keywords_list:
        pois = await fetch_poi_from_amap(city_code, keyword, POI_TYPES["酒店"], limit)
        all_hotels.extend(pois)
        if len(all_hotels) >= limit:
            break
    
    # 去重并限制数量
    seen = set()
    unique_hotels = []
    for poi in all_hotels:
        name = poi.get("name", "")
        if name not in seen:
            seen.add(name)
            unique_hotels.append(poi)
            if len(unique_hotels) >= limit:
                break
    
    price_range = BUDGET_PRICE_MAP.get(budget, BUDGET_PRICE_MAP["舒适"])
    
    return [{
        "name": poi.get("name", ""),
        "address": poi.get("address", ""),
        "type": poi.get("type", ""),
        "tel": poi.get("tel", ""),
        "location": poi.get("location", ""),
        "price_range": price_range,
        "distance": poi.get("distance", ""),
        "business_area": poi.get("business_area", "")
    } for poi in unique_hotels]

async def fetch_real_traffic_from_amap(city: str) -> Dict[str, Any]:
    """获取真实交通数据"""
    city_code = await get_city_code(city)
    if not city_code:
        return {}
    
    train_pois = await fetch_poi_from_amap(city_code, "火车站", POI_TYPES["火车站"], 2)
    airport_pois = await fetch_poi_from_amap(city_code, "机场", POI_TYPES["机场"], 1)
    
    return {
        "train_stations": [{
            "name": poi.get("name", ""),
            "address": poi.get("address", ""),
            "type": "火车站"
        } for poi in train_pois],
        "airports": [{
            "name": poi.get("name", ""),
            "address": poi.get("address", ""),
            "type": "机场"
        } for poi in airport_pois]
    }

# ==================== API 路由 ====================
@app.post("/api/generate-plan", response_model=PlanResponse)
async def generate_plan(params: PlanRequest):
    """生成AI旅行计划接口"""
    # 处理偏好参数
    prefs_str = "、".join(params.preferences) if params.preferences else "休闲观光"
    
    # 并发获取真实数据
    tasks = [
        fetch_real_spots_from_amap(params.destination),
        fetch_real_food_from_amap(params.destination),
        fetch_real_hotel_from_amap(params.destination, params.budget),
        fetch_real_traffic_from_amap(params.destination),
        get_real_weather(params.destination)
    ]
    spots_data, food_data, hotel_data, traffic_data, weather_str = await asyncio.gather(*tasks)
    
    # 构建数据摘要
    spots_summary = "\n".join([f"- {s.get('name','')} ({s.get('address','')})" 
                             for s in spots_data]) if spots_data else "暂无数据"
    food_summary = "\n".join([f"- {f.get('name','')} - {f.get('type','')}" 
                             for f in food_data]) if food_data else "暂无数据"
    hotel_summary = "\n".join([f"- {h.get('name','')} - {h.get('address','')} ({h.get('type','')}) - 价格：{h.get('price_range','')}" 
                              for h in hotel_data]) if hotel_data else "【无可用酒店】"
    
    # 酒店为空时的特殊提示
    hotel_constraint = ""
    if not hotel_data:
        hotel_constraint = """
【重要警告】可用酒店列表为空！必须遵守以下规则：
- 严禁编造任何酒店名称或地址！
- hotel字段必须设置为空字符串 ""
- 行程中的所有出发地和目的地都必须使用真实景点
- 格式示例："morning": "08:30 从住宿地出发 → 09:00 到达XX景点"
- 注意：使用"住宿地"而非具体酒店名称
"""
    
    train_names = [t.get("name", "") for t in traffic_data.get("train_stations", [])]
    airport_names = [a.get("name", "") for a in traffic_data.get("airports", [])]
    traffic_summary = f"火车站: {', '.join(train_names)}" if train_names else "暂无火车站数据"
    if airport_names:
        traffic_summary += f"\n机场: {', '.join(airport_names)}"

    # 计算预算分配
    budget_info = ""
    if params.total_budget:
        allocation = calculate_budget_allocation(params.total_budget, params.days, params.people_count)
        budget_info = f"""
【预算分配详情】
- 总预算：{params.total_budget}元
- 人均每日预算：{allocation['人均每日预算']}元
- 住宿预算（每晚/人）：{allocation['人均住宿预算（每晚）']}元（占总预算40%）
- 餐饮预算（每天/人）：{allocation['人均餐饮预算（每天）']}元（占总预算30%）
- 交通预算（每天/人）：{allocation['人均交通预算（每天）']}元（占总预算15%）
- 门票预算（每天/人）：{allocation['人均门票预算（每天）']}元（占总预算10%）
- 其他预算（每天/人）：{allocation['人均其他预算（每天）']}元（占总预算5%）
"""
    
    # 构建Prompt
    prompt = f"""【任务】请为{params.people_count}人规划{params.days}天的{params.destination}旅行计划

【基础参数】
- 目的地：{params.destination}
- 旅行天数：{params.days}天（必须生成{params.days}个行程日，不能多也不能少）
- 出行人数：{params.people_count}人
- 出发日期：{params.start_date or '近期'}
- 预算等级：{params.budget}
- 旅行偏好：{prefs_str}

【预算详情】
{budget_info if budget_info else f"预算等级：{params.budget}（经济档100-200元/晚/人，舒适档200-350元/晚/人，豪华档400-700元/晚/人）"}

【实时天气】（来自高德地图API）
{weather_str}

【可用景点列表】（必须从中选择，不得编造）
{spots_summary}

【可用餐厅列表】（必须从中选择，不得编造）
{food_summary}

【可用酒店列表】（必须从中选择，符合{params.budget}预算，不得编造）
{hotel_summary}
{hotel_constraint}
【交通信息】
{traffic_summary}

【强制约束条件】
=== 天数约束 ===
1. itinerary数组长度必须严格等于{params.days}
2. day字段必须从1开始连续递增（1, 2, ..., {params.days}）
3. 每个行程日必须包含完整的morning、afternoon、evening安排

=== 地点约束 ===
1. 景点必须从【可用景点列表】中选择
2. 餐厅必须从【可用餐厅列表】中选择
3. 酒店必须从【可用酒店列表】中选择，且符合{params.budget}预算
4. 同一景点不得在不同天数重复安排
5. 禁止编造任何地点名称
6. 【酒店一致性约束】酒店名称必须与地址对应：
   - hotel字段格式："酒店名称（地址）- 价格：XX元/晚/人"
   - 地址必须是{params.destination}市的真实地址
   - 所有行程活动中提到的"酒店"必须与hotel字段中的酒店名称完全一致
7. 【酒店位置约束】酒店地址必须根据行程合理规划：
   - 酒店应位于主要景点集中的区域或交通便利的地段（如地铁站附近、市中心商圈）
   - 酒店位置应方便前往第一天行程的第一个景点
   - 酒店地址应在所有景点的地理中心附近，减少整体交通时间
8. 【酒店一致性约束】整个行程必须使用同一个酒店：
   - 所有天数的hotel字段必须完全相同（酒店名称和地址都要一致）
   - 禁止每天更换不同的酒店
   - 必须从【可用酒店列表】中选择真实酒店名称和地址
9. 【酒店真实性约束】必须使用真实酒店数据：
   - hotel字段必须完全匹配【可用酒店列表】中的某一个酒店（名称和地址都要匹配）
   - 严禁编造任何酒店名称或地址
   - 酒店地址必须包含具体的街道和门牌号
10. 【严禁】以下内容不能作为景点出现在行程中：
   - 租借单车/电动车（如"租借哈啰单车"）
   - 购物行为、交通换乘点、便利店等生活服务设施
   这些内容只能在tip字段中作为交通建议提及

=== 时间约束 ===
1. 上午活动时间：08:00-12:00，下午活动时间：14:00-18:00，晚上活动时间：18:00-22:00
2. 每个活动必须标注具体时间点（如08:30、14:00）
3. 相邻活动间必须预留至少30分钟交通时间
4. 必须包含午餐和晚餐时间安排

=== 路线规划约束 ===
1. 每天行程必须从酒店出发开始，格式："08:30 从酒店出发 → 09:00 到达景点A"
2. 景点安排要考虑地理位置，避免来回折返，按区域顺序安排
3. 每个景点活动要包含"出发→到达"的导航信息
4. 每天行程最后必须返回酒店："21:00 返回酒店休息"

=== 格式约束 ===
1. 必须返回纯JSON格式，无任何多余文本
2. 禁止包含Markdown标记（如```json）
3. JSON结构必须包含itinerary数组
4. 每个行程项必须包含day、title、morning、afternoon、evening、food、hotel、tip字段
5. title必须包含"第N天"或"Day N"标识

【输出示例】
{{
  "itinerary": [
    {{
      "day": 1,
      "title": "第1天 · 城市初体验",
      "morning": "08:30 从酒店出发 → 09:00 到达景点A（地址）游玩 → 11:30 前往景点B",
      "afternoon": "14:00 从景点A出发 → 14:30 到达景点B（地址）游玩 → 17:30 前往餐厅",
      "evening": "18:00 从景点B出发 → 18:30 到达餐厅用餐 → 21:00 返回酒店休息",
      "food": "餐厅名称 - 推荐菜品（人均XX元）",
      "hotel": "酒店名称（地址）- 价格：XX元/晚/人",
      "tip": "今日天气：{weather_str}，建议穿着：XX。从酒店到景点A约XX分钟，景点A到景点B约XX分钟"
    }}
  ]
}}

【检查清单（输出前务必检查）】
1. ✓ itinerary数组长度是否等于{params.days}？
2. ✓ day字段是否从1连续递增到{params.days}？
3. ✓ 所有地点是否都来自提供的真实数据列表？
4. ✓ 是否有重复安排的景点？
5. ✓ 每个活动是否都有具体时间点？
6. ✓ 酒店价格是否符合{params.budget}预算范围？
7. ✓ 是否返回纯JSON格式，无任何多余文本？
8. ✓ 每天行程是否从酒店出发开始？
9. ✓ 每天行程是否返回酒店结束？
10. ✓ 景点安排是否考虑地理位置合理性（避免折返）？
11. ✓ 是否包含清晰的"出发→到达"导航信息？
12. ✓ hotel字段是否包含酒店名称和对应地址？
13. ✓ 行程活动中提到的"酒店"是否与hotel字段中的酒店名称一致？
14. ✓ 酒店地址是否位于景点集中区域或交通便利地段？
15. ✓ 酒店位置是否方便前往第一天行程的第一个景点？
16. ✓ 所有天数的酒店是否完全相同（名称和地址一致）？
17. ✓ 是否从可用酒店列表中选择了真实酒店？
18. ✓ hotel字段是否完全匹配可用酒店列表中的某一个？
19. ✓ 酒店地址是否包含具体街道和门牌号？"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_EXTENDED},
        {"role": "user", "content": prompt}
    ]

    # AI生成行程
    ai_content = await get_ai_response_async(messages)

    # 解析AI返回的JSON
    try:
        clean_content = ai_content.strip()
        # 清理可能存在的markdown标记
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:-3].strip()
        elif clean_content.startswith("```"):
            clean_content = clean_content[3:-3].strip()
        
        plan_json = json.loads(clean_content)
    except json.JSONDecodeError:
        logger.error(f"AI返回的JSON解析失败: {ai_content[:200]}...")
        plan_json = {"itinerary": []}

    return PlanResponse(
        status="success",
        message="AI 行程生成完毕！",
        data={
            "city": params.destination,
            "days": params.days,
            "weather": weather_str,
            "itinerary": plan_json.get("itinerary", []),
            "real_data": {
                "spots": spots_data,
                "foods": food_data,
                "hotels": hotel_data,
                "traffic": traffic_data
            }
        }
    )

@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "travel-planner-api", "version": "2.1.0"}

# ==================== 数据查询API ====================
@app.get("/api/spots", response_model=PlanResponse)
async def get_spots(city: str = "漳州市"):
    """获取景点数据接口"""
    real_spots = await fetch_real_spots_from_amap(city)
    
    if real_spots:
        return PlanResponse(
            status="success",
            message=f"成功获取{city}景点数据",
            data={"city": city, "spots": real_spots}
        )
    
    # 降级返回预设数据
    default_spots = {
        "成都": [
            {"name": "大熊猫繁育研究基地", "address": "四川省成都市成华区外北熊猫大道1375号", "type": "动物园"},
            {"name": "宽窄巷子", "address": "四川省成都市青羊区金河宾馆西侧", "type": "历史文化街区"},
            {"name": "都江堰景区", "address": "四川省成都市都江堰市公园路", "type": "水利工程"}
        ],
        "大理": [
            {"name": "洱海", "address": "云南省大理白族自治州大理市", "type": "自然景观"},
            {"name": "大理古城", "address": "云南省大理白族自治州大理市大理古城", "type": "历史文化街区"},
            {"name": "崇圣寺三塔", "address": "云南省大理白族自治州大理市崇圣寺三塔文化旅游区", "type": "古迹"}
        ],
        "厦门": [
            {"name": "鼓浪屿", "address": "福建省厦门市思明区", "type": "岛屿"},
            {"name": "环岛路", "address": "福建省厦门市思明区环岛路", "type": "海滨道路"},
            {"name": "曾厝垵", "address": "福建省厦门市思明区曾厝垵", "type": "文化创意区"}
        ],
        "西安": [
            {"name": "秦始皇兵马俑博物馆", "address": "陕西省西安市临潼区秦俑馆公路", "type": "博物馆"},
            {"name": "大唐不夜城", "address": "陕西省西安市雁塔区大雁塔南广场", "type": "步行街"},
            {"name": "西安城墙", "address": "陕西省西安市碑林区南大街2号", "type": "古迹"}
        ],
        "重庆": [
            {"name": "洪崖洞", "address": "重庆市渝中区嘉陵江滨江路88号", "type": "民俗街区"},
            {"name": "长江索道", "address": "重庆市渝中区新华路153号", "type": "交通工具"},
            {"name": "磁器口古镇", "address": "重庆市沙坪坝区磁器口古镇", "type": "历史文化街区"}
        ]
    }
    return PlanResponse(
        status="success",
        message=f"返回{city}预设景点数据",
        data={"city": city, "spots": default_spots.get(city, default_spots["成都"])}
    )

@app.get("/api/weather", response_model=PlanResponse)
async def get_weather(city: str = "漳州市"):
    """获取真实天气数据"""
    weather = await get_real_weather(city)
    return PlanResponse(
        status="success",
        message="获取天气成功",
        data={"city": city, "weather": weather}
    )

@app.get("/api/food", response_model=PlanResponse)
async def get_food(city: str = "漳州市"):
    """获取真实美食数据"""
    real_food = await fetch_real_food_from_amap(city)
    
    if real_food:
        return PlanResponse(
            status="success",
            message=f"成功获取{city}美食数据",
            data={"city": city, "foods": real_food}
        )
    
    # 降级返回预设美食数据
    default_food = {
        "成都": [
            {"name": "小龙坎火锅", "address": "成都市锦江区大慈寺路", "type": "川味火锅", "price_range": "人均150元"},
            {"name": "玉林串串香", "address": "成都市武侯区玉林路", "type": "串串香", "price_range": "人均60元"},
            {"name": "蜀大侠火锅", "address": "成都市成华区建设路", "type": "川味火锅", "price_range": "人均100元"}
        ],
        "大理": [
            {"name": "苍洱春饭店", "address": "大理古城人民路", "type": "白族菜", "price_range": "人均50元"},
            {"name": "益恒饭店", "address": "大理古城复兴路", "type": "白族菜", "price_range": "人均45元"},
            {"name": "88号西点店", "address": "大理古城人民路", "type": "甜品咖啡", "price_range": "人均40元"}
        ],
        "厦门": [
            {"name": "中山路沙茶面", "address": "厦门市思明区中山路", "type": "沙茶面", "price_range": "人均25元"},
            {"name": "1980烧肉粽", "address": "厦门市思明区中山路", "type": "闽南小吃", "price_range": "人均30元"},
            {"name": "黄则和花生汤", "address": "厦门市思明区中山路", "type": "甜品", "price_range": "人均20元"}
        ]
    }
    return PlanResponse(
        status="success",
        message=f"返回{city}预设美食数据",
        data={"city": city, "foods": default_food.get(city, default_food["成都"])}
    )

@app.get("/api/hotel", response_model=PlanResponse)
async def get_hotel(
    city: str = "漳州市", 
    budget: str = "舒适",
    keyword: str = "",
    star: str = "",
    limit: int = 10
):
    """获取真实酒店数据
    
    参数:
    - city: 城市名称
    - budget: 预算等级（经济/舒适/豪华）
    - keyword: 搜索关键词
    - star: 星级筛选（如：三星级、四星级、五星级）
    - limit: 返回数量限制
    """
    logger.info(f"🔍 搜索酒店: 城市={city}, 预算={budget}, 关键词={keyword}, 星级={star}, 数量={limit}")
    
    real_hotels = await fetch_real_hotel_from_amap(city, budget, limit)
    
    # 如果有额外关键词筛选
    if keyword:
        keyword_lower = keyword.lower()
        real_hotels = [h for h in real_hotels if keyword_lower in h.get("name", "").lower() or keyword_lower in h.get("address", "").lower()]
    
    # 如果有星级筛选
    if star:
        star_lower = star.lower()
        real_hotels = [h for h in real_hotels if star_lower in h.get("type", "").lower()]
    
    if real_hotels:
        return PlanResponse(
            status="success",
            message=f"成功获取{city}酒店数据，共{len(real_hotels)}家",
            data={"city": city, "budget": budget, "hotels": real_hotels}
        )
    
    # 降级返回预设酒店数据
    default_hotels = {
        "经济": {
            "成都": [
                {"name": "成都梦之旅青年旅舍", "address": "成都市武侯区武侯祠大街", "type": "青年旅舍", "price_range": "50-80元/晚/人"},
                {"name": "瓦舍旅行酒店", "address": "成都市锦江区镗钯街", "type": "经济型酒店", "price_range": "100-150元/晚/人"},
                {"name": "7天连锁酒店", "address": "成都市青羊区人民中路", "type": "连锁酒店", "price_range": "80-120元/晚/人"}
            ],
            "厦门": [
                {"name": "鼓浪屿国际青年旅舍", "address": "厦门市思明区鼓浪屿", "type": "青年旅舍", "price_range": "60-100元/晚/人"},
                {"name": "如家快捷酒店", "address": "厦门市思明区中山路", "type": "连锁酒店", "price_range": "100-150元/晚/人"}
            ],
            "三亚": [
                {"name": "三亚湾青年旅舍", "address": "三亚市天涯区三亚湾", "type": "青年旅舍", "price_range": "50-80元/晚/人"},
                {"name": "城市便捷酒店", "address": "三亚市吉阳区", "type": "经济型酒店", "price_range": "120-180元/晚/人"}
            ]
        },
        "舒适": {
            "成都": [
                {"name": "成都凯宾斯基饭店", "address": "成都市武侯区人民南路", "type": "四星级酒店", "price_range": "300-500元/晚/人"},
                {"name": "成都JW万豪酒店", "address": "成都市锦江区东大街", "type": "五星级酒店", "price_range": "600-800元/晚/人"},
                {"name": "成都富力丽思卡尔顿酒店", "address": "成都市青羊区顺城大街", "type": "五星级酒店", "price_range": "700-900元/晚/人"}
            ],
            "厦门": [
                {"name": "厦门海悦山庄酒店", "address": "厦门市思明区环岛南路", "type": "五星级酒店", "price_range": "500-700元/晚/人"},
                {"name": "厦门香格里拉大酒店", "address": "厦门市思明区观音山", "type": "五星级酒店", "price_range": "600-800元/晚/人"}
            ],
            "三亚": [
                {"name": "三亚亚龙湾喜来登度假酒店", "address": "三亚市吉阳区亚龙湾", "type": "五星级酒店", "price_range": "800-1200元/晚/人"},
                {"name": "三亚海棠湾洲际度假酒店", "address": "三亚市海棠湾区", "type": "五星级酒店", "price_range": "1000-1500元/晚/人"}
            ]
        },
        "豪华": {
            "成都": [
                {"name": "成都瑞吉酒店", "address": "成都市青羊区太升南路", "type": "豪华五星级酒店", "price_range": "1500-2500元/晚/人"},
                {"name": "成都钓鱼台精品酒店", "address": "成都市青羊区宽窄巷子", "type": "奢华精品酒店", "price_range": "2000-3000元/晚/人"},
                {"name": "成都华尔道夫酒店", "address": "成都市高新区", "type": "豪华五星级酒店", "price_range": "1800-2800元/晚/人"}
            ],
            "厦门": [
                {"name": "厦门鼓浪屿菽庄花园酒店", "address": "厦门市思明区鼓浪屿", "type": "奢华精品酒店", "price_range": "2000-3000元/晚/人"},
                {"name": "厦门七尚酒店", "address": "厦门市湖里区", "type": "豪华五星级酒店", "price_range": "1500-2500元/晚/人"}
            ],
            "三亚": [
                {"name": "三亚亚特兰蒂斯酒店", "address": "三亚市海棠湾区", "type": "豪华度假酒店", "price_range": "2000-3500元/晚/人"},
                {"name": "三亚蜈支洲岛珊瑚酒店", "address": "三亚市海棠湾区蜈支洲岛", "type": "奢华海岛酒店", "price_range": "3000-5000元/晚/人"}
            ]
        }
    }
    
    hotels = default_hotels.get(budget, default_hotels["舒适"]).get(city, default_hotels["舒适"]["成都"])
    
    # 对预设数据也应用筛选
    if keyword:
        keyword_lower = keyword.lower()
        hotels = [h for h in hotels if keyword_lower in h.get("name", "").lower()]
    if star:
        star_lower = star.lower()
        hotels = [h for h in hotels if star_lower in h.get("type", "").lower()]
    
    return PlanResponse(
        status="success",
        message=f"返回{city}预设酒店数据",
        data={"city": city, "budget": budget, "hotels": hotels}
    )

@app.get("/api/traffic", response_model=PlanResponse)
async def get_traffic(city: str = "漳州市"):
    """获取真实交通数据"""
    real_traffic = await fetch_real_traffic_from_amap(city)
    
    if real_traffic:
        return PlanResponse(
            status="success",
            message=f"成功获取{city}交通数据",
            data={"city": city, **real_traffic}
        )
    
    # 降级返回预设交通数据
    default_traffic = {
        "成都": {
            "train_stations": [
                {"name": "成都站（火车北站）", "address": "成都市金牛区二环路"},
                {"name": "成都东站", "address": "成都市成华区保和镇"},
                {"name": "成都西站", "address": "成都市青羊区苏坡街道"}
            ],
            "airports": [{"name": "成都双流国际机场", "address": "成都市双流区"}],
            "city_center": "104.0657,30.6598"
        }
    }
    return PlanResponse(
        status="success",
        message=f"返回{city}预设交通数据",
        data={"city": city, **default_traffic.get(city, default_traffic["成都"])}
    )

@app.get("/api/city-data", response_model=PlanResponse)
async def get_city_data(city: str = "漳州市", budget: str = "舒适"):
    """获取城市完整数据（景点+美食+酒店+交通）"""
    tasks = [
        fetch_real_spots_from_amap(city),
        fetch_real_food_from_amap(city),
        fetch_real_hotel_from_amap(city, budget),
        fetch_real_traffic_from_amap(city),
        get_real_weather(city)
    ]
    spots, foods, hotels, traffic, weather = await asyncio.gather(*tasks)
    
    return PlanResponse(
        status="success",
        message=f"成功获取{city}完整数据",
        data={
            "city": city,
            "weather": weather,
            "spots": spots,
            "foods": foods,
            "hotels": hotels,
            "traffic": traffic
        }
    )

@app.get("/api/city-ranking", response_model=PlanResponse)
async def get_city_ranking():
    """获取热门城市排名数据"""
    ranking = [
        {"rank": 1, "city": "成都", "feature": "🐼 熊猫基地", "hot": 98000},
        {"rank": 2, "city": "大理", "feature": "🌊 洱海", "hot": 82000},
        {"rank": 3, "city": "厦门", "feature": "🏖️ 鼓浪屿", "hot": 75000},
        {"rank": 4, "city": "西安", "feature": "🏯 兵马俑", "hot": 69000},
        {"rank": 5, "city": "重庆", "feature": "🌉 洪崖洞", "hot": 62000},
        {"rank": 6, "city": "三亚", "feature": "🏝️ 亚龙湾", "hot": 58000},
        {"rank": 7, "city": "伊犁", "feature": "🌿 那拉提草原", "hot": 45000},
        {"rank": 8, "city": "拉萨", "feature": "⛰️ 布达拉宫", "hot": 42000}
    ]
    return PlanResponse(
        status="success",
        message="获取城市排名成功",
        data={"ranking": ranking}
    )

@app.get("/api/stories", response_model=PlanResponse)
async def get_stories():
    """获取旅行故事数据"""
    stories = [
        {
            "id": 1,
            "title": "洱海骑行日记",
            "author": "风一样的旅人",
            "location": "大理",
            "cover": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=500",
            "content": "在洱海边骑车，风里有自由的味道。",
            "fullContent": "清晨从古城出发，租了一辆小电驴沿着环海西路骑行。天空蓝得像油画，路过喜洲古镇吃了破酥粑粑。傍晚在双廊看日落，金色的光洒在水面上，那一刻觉得人间值得。",
            "images": ["https://images.unsplash.com/photo-1501785888041-af3ef285b470?w=800", "https://images.unsplash.com/photo-1528543606781-2f6e6857f318?w=800"],
            "tags": ["治愈", "骑行"]
        },
        {
            "id": 2,
            "title": "成都慢生活",
            "author": "吃客不胖",
            "location": "成都",
            "cover": "https://images.unsplash.com/photo-1564349683136-77e08dba1ef7?w=500",
            "content": "茶馆、火锅、大熊猫，巴适得板！",
            "fullContent": "成都的节奏太舒服了，人民公园鹤鸣茶社点一杯盖碗茶，掏耳朵。大熊猫基地看幼崽翻滚。晚上吃火锅，一定要点鸭肠和毛肚。",
            "images": ["https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=800", "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=800"],
            "tags": ["美食", "休闲"]
        },
        {
            "id": 3,
            "title": "西安古都寻梦",
            "author": "长安客",
            "location": "西安",
            "cover": "https://images.unsplash.com/photo-1533104816931-20fa691ff6ca?w=500",
            "content": "兵马俑震撼，不夜城璀璨。",
            "fullContent": "兵马俑一定要请讲解，否则就是看泥人。大雁塔北广场音乐喷泉壮观。晚上大唐不夜城仿佛穿越，汉服小姐姐很多。",
            "images": ["https://images.unsplash.com/photo-1545569341-9eb8b30979d9?w=800", "https://images.unsplash.com/photo-1508804185872-d7badad00f7d?w=800"],
            "tags": ["历史", "美食"]
        },
        {
            "id": 4,
            "title": "重庆魔幻8D",
            "author": "山城棒棒",
            "location": "重庆",
            "cover": "https://images.unsplash.com/photo-1569154941061-e231b4725ef1?w=500",
            "content": "洪崖洞像千与千寻，轻轨穿楼。",
            "fullContent": "导航没用，问路更靠谱。洪崖洞夜景绝美，但人巨多。李子坝轻轨穿楼打卡。南山一棵树看全城。",
            "images": ["https://images.unsplash.com/photo-1585409677983-0f6c41ca9c3b?w=800", "https://images.unsplash.com/photo-1504198453319-5ce911bafcde?w=800"],
            "tags": ["魔幻", "美食"]
        },
        {
            "id": 5,
            "title": "三亚蓝色假期",
            "author": "海浪声声",
            "location": "三亚",
            "cover": "https://images.unsplash.com/photo-1540541338287-41700207dee6?w=500",
            "content": "亚龙湾的沙细如粉，蜈支洲岛潜水绝佳。",
            "fullContent": "亚龙湾的沙质洁白细腻，海水湛蓝。蜈支洲岛潜水能看到珊瑚和鱼群。鹿回头俯瞰三亚湾夜景，浪漫至极。",
            "images": ["https://images.unsplash.com/photo-1505228395891-9a51e7e86bf6?w=800", "https://images.unsplash.com/photo-1518548419970-58e3b4079ab2?w=800"],
            "tags": ["海岛", "潜水"]
        },
        {
            "id": 6,
            "title": "厦门文艺之旅",
            "author": "海风日记",
            "location": "厦门",
            "cover": "https://images.unsplash.com/photo-1551632811-561732d1e306?w=500",
            "content": "鼓浪屿的小巷，沙坡尾的咖啡馆。",
            "fullContent": "在鼓浪屿住了一晚，清晨没有游客的时候最美。日光岩俯瞰全岛，菽庄花园精巧。沙坡尾很多设计师店铺，艺术西区适合拍照。",
            "images": ["https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800", "https://images.unsplash.com/photo-1472214103451-9374bd1c798c?w=800"],
            "tags": ["文艺", "海岛"]
        }
    ]
    return PlanResponse(
        status="success",
        message="获取旅行故事成功",
        data={"stories": stories}
    )

# ==================== 路线规划API ====================
class RoutePoint(BaseModel):
    """路线节点"""
    name: str
    address: Optional[str] = None
    type: Optional[str] = None  # 景点、餐厅、酒店

class RouteRequest(BaseModel):
    """路线规划请求"""
    city: str
    points: Optional[List[RoutePoint]] = None
    # 可选：直接传入 AI 已生成的 itinerary（更结构化）
    itinerary: Optional[List[Dict[str, Any]]] = None


# 小工具：从字符串中抽取可能的地点词
# 优先匹配「景点、博物馆、公园、广场、大厦、寺、街、巷、路、塔、桥、湖、山、店、楼、中心、美食、餐厅、酒店」等关键词
import re as _re

_LOCATION_KEYWORDS = [
    r"[\u4e00-\u9fa5A-Za-z0-9]+博物馆", r"[\u4e00-\u9fa5A-Za-z0-9]+纪念馆",
    r"[\u4e00-\u9fa5A-Za-z0-9]+公园", r"[\u4e00-\u9fa5A-Za-z0-9]+广场",
    r"[\u4e00-\u9fa5A-Za-z0-9]+大厦", r"[\u4e00-\u9fa5A-Za-z0-9]+中心",
    r"[\u4e00-\u9fa5A-Za-z0-9]+寺", r"[\u4e00-\u9fa5A-Za-z0-9]+塔",
    r"[\u4e00-\u9fa5A-Za-z0-9]+桥", r"[\u4e00-\u9fa5A-Za-z0-9]+湖",
    r"[\u4e00-\u9fa5A-Za-z0-9]+山", r"[\u4e00-\u9fa5A-Za-z0-9]+楼",
    r"[\u4e00-\u9fa5A-Za-z0-9]+街", r"[\u4e00-\u9fa5A-Za-z0-9]+巷",
    r"[\u4e00-\u9fa5A-Za-z0-9]+路", r"[\u4e00-\u9fa5A-Za-z0-9]+道",
    r"[\u4e00-\u9fa5A-Za-z0-9]+度假村", r"[\u4e00-\u9fa5A-Za-z0-9]+景区",
    r"[\u4e00-\u9fa5A-Za-z0-9]+名胜", r"[\u4e00-\u9fa5A-Za-z0-9]+岛",
    r"[\u4e00-\u9fa5A-Za-z0-9]+酒店", r"[\u4e00-\u9fa5A-Za-z0-9]+宾馆",
    r"[\u4e00-\u9fa5A-Za-z0-9]+餐厅", r"[\u4e00-\u9fa5A-Za-z0-9]+美食",
    r"[\u4e00-\u9fa5A-Za-z0-9]+小吃", r"[\u4e00-\u9fa5A-Za-z0-9]+店",
]
_STOP_WORDS = {"早餐", "午餐", "晚餐", "下午茶", "夜宵", "休息", "自由活动",
               "出发", "前往", "到达", "游览", "参观", "打卡", "游玩",
               "建议", "时间", "开放", "门票", "公里", "分钟", "小时"}


def _extract_location_names(text: str) -> List[str]:
    """从一段自然语言文本中抽取可能的地点/景点名称。"""
    if not text:
        return []
    cleaned = text.replace("\n", " ")
    found = []
    for pattern in _LOCATION_KEYWORDS:
        matches = _re.findall(pattern, cleaned)
        for m in matches:
            m = m.strip()
            if len(m) < 2:
                continue
            if any(sw in m for sw in _STOP_WORDS):
                continue
            if m not in found:
                found.append(m)
    # 去重但保持顺序
    uniq = []
    for n in found:
        if n not in uniq:
            uniq.append(n)
    return uniq[:6]  # 每天最多取 6 个地点，避免过多


async def _geocode_with_city(name: str, city: str) -> Optional[Dict[str, Any]]:
    """对单个地点进行地理编码，尝试 名称+城市，再回退到纯名称。"""
    if not name:
        return None
    # 先尝试：城市 + 名称
    coords = await fetch_location_from_address(f"{city} {name}")
    if not coords and city not in name:
        # 再试一次：纯名称
        coords = await fetch_location_from_address(name)
    if not coords:
        return None
    return {"lng": coords["lng"], "lat": coords["lat"]}


@app.post("/api/route-planning", response_model=PlanResponse)
async def plan_route(request: RouteRequest):
    """根据景点列表或行程规划，返回每个地点的真实坐标与路线信息。

    两种调用方式：
    1) 传入 points: [{name, address, type}, ...]
    2) 传入 itinerary（AI 返回的完整日程数组，含 morning/afternoon/evening/food/hotel）
    """
    city = (request.city or "").strip()
    collected: List[Dict[str, Any]] = []
    seen_names: set = set()

    # --- 方式 A：直接传入 points ---
    if request.points:
        tasks = []
        for p in request.points:
            key = (p.name or p.address or "").strip()
            if not key or key in seen_names:
                continue
            seen_names.add(key)
            tasks.append((key, p.type or "景点"))
        if tasks:
            coords_list = await asyncio.gather(
                *[_geocode_with_city(name, city) for name, _ in tasks]
            )
            for (name, ptype), coords in zip(tasks, coords_list):
                collected.append({
                    "name": name,
                    "type": ptype,
                    "coords": coords,
                    "day": 0,
                    "part": "points"
                })

    # --- 方式 B：传入 AI 行程 itinerary（主要使用路径）---
    if request.itinerary:
        for day_item in request.itinerary:
            day_index = int(day_item.get("day") or 1)
            parts_map = {
                "morning": "上午",
                "afternoon": "下午",
                "evening": "晚上",
                "food": "美食",
                "hotel": "住宿",
            }
            for key, label in parts_map.items():
                content = day_item.get(key)
                if not content:
                    continue
                # food / hotel 是整段描述，直接当地点描述处理
                if key in ("food", "hotel"):
                    candidates = [content]
                    ptype = "餐厅" if key == "food" else "酒店"
                else:
                    candidates = _extract_location_names(str(content))
                    ptype = "景点"
                for name in candidates:
                    name = str(name).strip()
                    if len(name) < 2 or name in seen_names:
                        continue
                    seen_names.add(name)
                    collected.append({
                        "name": name,
                        "type": ptype,
                        "coords": None,  # 下面统一批量获取
                        "day": day_index,
                        "part": label,
                    })

    # 批量地理编码
    if collected:
        coords_results = await asyncio.gather(
            *[_geocode_with_city(item["name"], city) for item in collected]
        )
        for item, coords in zip(collected, coords_results):
            item["coords"] = coords

    # 统计有效坐标数量
    valid_count = sum(1 for c in collected if c.get("coords"))

    return PlanResponse(
        status="success",
        message=f"路线规划完成：共 {len(collected)} 个地点，成功定位 {valid_count} 个",
        data={
            "city": city,
            "waypoints": collected,
            "total": len(collected),
            "valid": valid_count,
        }
    )

# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn
    logger.info("服务启动: http://localhost:8000")
    logger.info("接口文档: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
