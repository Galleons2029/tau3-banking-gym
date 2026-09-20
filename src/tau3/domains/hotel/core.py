"""Deterministic fictional hotel simulator; Python standard library only."""
from copy import deepcopy
from datetime import date, datetime, timedelta
import json

ROOMS = {
    'king': {'rate': 500, 'capacity': 2, 'accessible': False},
    'twin': {'rate': 450, 'capacity': 2, 'accessible': False},
    'family': {'rate': 800, 'capacity': 3, 'accessible': False},
    'accessible': {'rate': 550, 'capacity': 2, 'accessible': True},
}

def dates(start, end):
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    if b <= a or (b-a).days > 30:
        raise ValueError('无效住宿日期')
    return [(a+timedelta(days=i)).isoformat() for i in range((b-a).days)]

def canonical_state(state):
    """Generated service IDs are bookkeeping, not business outcomes; retain multiplicity."""
    result=deepcopy(state)
    result['services']=sorted(result['services'].values(),key=lambda v:json.dumps(v,sort_keys=True))
    return result

class Hotel:
    def __init__(self, state):
        self.state = deepcopy(state)

    def get_guest(self, guest_id: str) -> dict:
        """查询客人资料；guest_id 由用户提供。"""
        return {k:deepcopy(v) for k,v in self.state['guests'][guest_id].items() if k!='verification_code'}

    def _rate(self, room):
        return self.state.get('room_rates',{}).get(room,ROOMS[room]['rate'])

    def get_hotel_info(self) -> dict:
        """查询当前酒店时间、各房型本场景每晚价格；覆盖通用政策中的默认房价。"""
        return {'now':self.state['now'],'hotel':self.state.get('hotel_name','澄湾酒店'),
                'rooms':{r:{**p,'rate':self._rate(r)} for r,p in ROOMS.items()}}

    def get_reservation(self, guest_id: str, reservation_id: str) -> dict:
        """查询指定客人名下订单；不匹配时拒绝。"""
        r = self.state['reservations'][reservation_id]
        if r['guest_id'] != guest_id:
            raise ValueError('订单不属于该客人')
        return deepcopy(r)

    def search_rooms(self, check_in: str, check_out: str, guests: int, accessible: bool = False) -> list:
        """查询整个日期区间有库存且满足人数/无障碍要求的房型和总价。"""
        self._future(check_in)
        nights = dates(check_in, check_out)
        if type(guests) is not int or guests < 1:
            raise ValueError('人数无效')
        result = []
        for room, p in ROOMS.items():
            if guests <= p['capacity'] and (not accessible or p['accessible']) and self._available(room, nights):
                result.append({'room': room, **p, 'rate':self._rate(room),'total': self._rate(room)*len(nights)})
        return result

    def _future(self, start):
        if date.fromisoformat(start) < datetime.fromisoformat(self.state['now']).date():
            raise ValueError('不能预订过去日期')

    def _available(self, room, nights, exclude=None):
        for day in nights:
            used = sum(r['room'] == room and r['status'] in ('booked', 'checked_in') and
                       r['check_in'] <= day < r['check_out'] for key, r in self.state['reservations'].items() if key != exclude)
            if self.state['inventory'].get(day, {}).get(room, 0) <= used:
                return False
        return True

    @staticmethod
    def _confirm(confirmed):
        if confirmed is not True:
            raise ValueError('需要用户明确确认')

    def book_room(self, guest_id: str, room: str, check_in: str, check_out: str, guests: int, rate_plan: str, confirmed: bool) -> dict:
        """全额预付预订一间房；价型 flexible/nonrefundable。先向用户确认完整方案。"""
        self._confirm(confirmed)
        self.get_guest(guest_id)
        if rate_plan not in ('flexible', 'nonrefundable'):
            raise ValueError('价型无效')
        options = self.search_rooms(check_in, check_out, guests)
        option = next((p for p in options if p['room'] == room), None)
        if option is None:
            raise ValueError('无满足条件库存')
        rid = 'new_' + str(len(self.state['reservations'])+1)
        r = dict(guest_id=guest_id, room=room, check_in=check_in, check_out=check_out,
                 guests=guests, rate_plan=rate_plan, status='booked', paid=option['total'],
                 refunded=0, breakfast=False, checkout_hour=12, late_available=True)
        self.state['reservations'][rid] = r
        return {'reservation_id': rid, **deepcopy(r)}

    def modify_reservation(self, guest_id: str, reservation_id: str, room: str, check_in: str, check_out: str, confirmed: bool) -> dict:
        """改 flexible 未入住订单的日期/房型，差额记入付款或退款；目标库存不足时不修改。"""
        self._confirm(confirmed)
        r = self.get_reservation(guest_id, reservation_id)
        self._future(check_in)
        nights = dates(check_in, check_out)
        if r['status'] != 'booked' or r['rate_plan'] != 'flexible':
            raise ValueError('该订单不可修改')
        if r.get('channel','direct')!='direct':raise ValueError('第三方渠道订单需要渠道处理')
        if room not in ROOMS or r['guests'] > ROOMS[room]['capacity'] or not self._available(room, nights, reservation_id):
            raise ValueError('目标库存或人数条件不满足')
        total = self._rate(room)*len(nights)
        delta = total-(r['paid']-r['refunded'])
        r.update(room=room, check_in=check_in, check_out=check_out)
        r['paid'] += max(0, delta)
        r['refunded'] += max(0, -delta)
        self.state['reservations'][reservation_id] = r
        return deepcopy(r)

    def quote_cancel(self, guest_id: str, reservation_id: str) -> dict:
        """返回取消退款与扣费；48小时按入住日当地00:00计算。"""
        r = self.get_reservation(guest_id, reservation_id)
        if r['status'] != 'booked':
            raise ValueError('仅未入住订单可取消')
        if r.get('channel','direct')!='direct':raise ValueError('第三方渠道订单需要渠道处理')
        balance = r['paid']-r['refunded']
        start = datetime.fromisoformat(r['check_in']+'T00:00:00+08:00')
        early = start-datetime.fromisoformat(self.state['now']) >= timedelta(hours=48)
        refund = 0 if r['rate_plan']=='nonrefundable' else balance if early else max(0,balance-self._rate(r['room']))
        return {'refund': refund, 'fee': balance-refund}

    def cancel_reservation(self, guest_id: str, reservation_id: str, confirmed: bool) -> dict:
        """用户确认退款金额后取消预订；退款写入原订单付款记录。"""
        self._confirm(confirmed)
        quote = self.quote_cancel(guest_id, reservation_id)
        r = self.state['reservations'][reservation_id]
        r['refunded'] += quote['refund']
        r['status'] = 'cancelled'
        return deepcopy(r)

    def _inhouse(self, guest_id, reservation_id, confirmed):
        self._confirm(confirmed)
        r = self.get_reservation(guest_id, reservation_id)
        if r['status'] != 'checked_in':
            raise ValueError('仅在店订单可办理')
        return r

    def add_breakfast(self, guest_id: str, reservation_id: str, confirmed: bool) -> dict:
        """为在店订单全住期全住客购买早餐，每人每晚60元；不可重复购买。"""
        r = self._inhouse(guest_id, reservation_id, confirmed)
        if r['breakfast']:
            raise ValueError('已含早餐')
        r['paid'] += 60*r['guests']*len(dates(r['check_in'],r['check_out']))
        r['breakfast'] = True
        self.state['reservations'][reservation_id] = r
        return deepcopy(r)

    def late_checkout(self, guest_id: str, reservation_id: str, hour: int, confirmed: bool) -> dict:
        """在店延迟退房，仅14/16点；gold到14点免费，其余收半晚。"""
        r = self._inhouse(guest_id, reservation_id, confirmed)
        if hour not in (14,16) or not r['late_available'] or r['checkout_hour'] != 12:
            raise ValueError('无法办理延迟退房')
        free = hour == 14 and self.get_guest(guest_id)['tier'] == 'gold'
        r['paid'] += 0 if free else self._rate(r['room'])//2
        r['checkout_hour'] = hour
        self.state['reservations'][reservation_id] = r
        return deepcopy(r)

    def create_service(self, guest_id: str, reservation_id: str, category: str, confirmed: bool) -> dict:
        """创建在店维修工单 ac/noise/plumbing；相同类别未关闭工单不重复创建。"""
        self._inhouse(guest_id, reservation_id, confirmed)
        if category not in ('ac','noise','plumbing'):
            raise ValueError('类别无效')
        if any(t['reservation_id']==reservation_id and t['category']==category and t['status']=='open' for t in self.state['services'].values()):
            raise ValueError('已有未关闭同类工单')
        sid = 'service_'+str(len(self.state['services'])+1)
        self.state['services'][sid] = dict(reservation_id=reservation_id, category=category, status='open')
        return {'service_id':sid, **self.state['services'][sid]}

    def issue_invoice(self, guest_id: str, reservation_id: str, title: str, amount: int, confirmed: bool) -> dict:
        """退房后开个人发票，金额等于实付净额，抬头须用户提供，每订单一张。"""
        self._confirm(confirmed)
        r = self.get_reservation(guest_id,reservation_id)
        if r['status'] != 'checked_out' or type(amount) is not int or amount != r['paid']-r['refunded'] or not title.strip():
            raise ValueError('发票条件不满足')
        if reservation_id in self.state['invoices']:
            raise ValueError('已开发票')
        self.state['invoices'][reservation_id] = {'title':title,'amount':amount}
        return deepcopy(self.state['invoices'][reservation_id])

    def list_reservations(self, guest_id: str) -> dict:
        """列出客人本人的全部订单；需要先向客人取得guest_id。"""
        self.get_guest(guest_id)
        return {k:deepcopy(v) for k,v in self.state['reservations'].items() if v['guest_id']==guest_id}

    def quote_modification(self, guest_id: str, reservation_id: str, room: str, check_in: str, check_out: str) -> dict:
        """只读改期报价，排除原订单占用；返回total/difference/available，不修改状态。"""
        r=self.get_reservation(guest_id,reservation_id)
        self._future(check_in)
        nights=dates(check_in,check_out)
        if room not in ROOMS or r['status']!='booked' or r['rate_plan']!='flexible':
            raise ValueError('该订单不可修改')
        if r.get('channel','direct')!='direct':raise ValueError('第三方渠道订单需要渠道处理')
        total=self._rate(room)*len(nights)
        return {'total':total,'difference':total-(r['paid']-r['refunded']),
                'available':r['guests']<=ROOMS[room]['capacity'] and self._available(room,nights,reservation_id)}

    def get_folio(self, guest_id: str, reservation_id: str) -> dict:
        """查询订单未结消费明细和净实付。每项含amount、paid；paid=false需要结算。"""
        r=self.get_reservation(guest_id,reservation_id)
        charges=deepcopy(self.state.get('folios',{}).get(reservation_id,{}))
        return {'charges':charges,'due':sum(c['amount'] for c in charges.values() if not c['paid']),
                'net_paid':r['paid']-r['refunded']}

    def settle_folio(self, guest_id: str, reservation_id: str, amount: int, confirmed: bool) -> dict:
        """结清所有未结消费；金额必须精确等于due且大于0，模拟原支付方式扣款。"""
        r=self._inhouse(guest_id,reservation_id,confirmed)
        due=self.get_folio(guest_id,reservation_id)['due']
        if type(amount) is not int or amount!=due or due<=0:
            raise ValueError('结算金额与未结消费不符')
        r['paid']+=due
        self.state['reservations'][reservation_id]=r
        for charge in self.state['folios'][reservation_id].values():charge['paid']=True
        return deepcopy(r)

    def check_out(self, guest_id: str, reservation_id: str, confirmed: bool) -> dict:
        """在计划退房日或之后办理退房，必须先结清消费；退房后不能再购买在店服务。"""
        r=self._inhouse(guest_id,reservation_id,confirmed)
        if datetime.fromisoformat(self.state['now']).date()<date.fromisoformat(r['check_out']):
            raise ValueError('本领域不支持提前退房')
        if self.get_folio(guest_id,reservation_id)['due']:
            raise ValueError('请先结清未付消费')
        r['status']='checked_out'
        self.state['reservations'][reservation_id]=r
        return deepcopy(r)

    def get_services(self, guest_id: str, reservation_id: str) -> dict:
        """查询本人订单工单；补偿只根据verified_failure和credited标记判断。"""
        self.get_reservation(guest_id,reservation_id)
        return {k:deepcopy(v) for k,v in self.state['services'].items() if v['reservation_id']==reservation_id}

    def cancel_service(self, guest_id: str, reservation_id: str, service_id: str, confirmed: bool) -> dict:
        """客人明确撤回后取消自己的open工单，不可取消其他订单或已关闭工单。"""
        self._inhouse(guest_id,reservation_id,confirmed)
        t=self.get_services(guest_id,reservation_id).get(service_id)
        if t is None or t['status']!='open':raise ValueError('工单不能取消')
        self.state['services'][service_id]['status']='cancelled'
        return deepcopy(self.state['services'][service_id])

    def issue_service_credit(self, guest_id: str, reservation_id: str, service_id: str, amount: int, confirmed: bool) -> dict:
        """核实维修失败后退100元，每工单一次；仅在店且未开发票，无自动升级或额外补偿。"""
        r=self._inhouse(guest_id,reservation_id,confirmed)
        t=self.get_services(guest_id,reservation_id).get(service_id)
        if (not t or not t.get('verified_failure',False) or t.get('credited',False)
            or type(amount) is not int or amount!=100 or r['paid']-r['refunded']<100
            or reservation_id in self.state['invoices']):
            raise ValueError('补偿资格或金额不满足')
        self.state['services'][service_id]['credited']=True
        self.state['reservations'][reservation_id]['refunded']+=100
        return deepcopy(self.state['reservations'][reservation_id])

    def search_shuttles(self, service_date: str, seats: int, bags: int, accessible: bool = False) -> list:
        """查询酒店到机场班车；所有行李含随身件，返回剩余座位/行李位和总价。"""
        if type(seats) is not int or type(bags) is not int or seats<1 or bags<0:
            raise ValueError('人数/行李件数无效')
        now=datetime.fromisoformat(self.state['now']); result=[]
        for sid,slot in self.state.get('shuttle_slots',{}).items():
            departure=datetime.fromisoformat(slot['departure'])
            reservations=[b for b in self.state.get('shuttle_bookings',{}).values() if b['slot_id']==sid and b['status']=='booked']
            free_seats=slot['seats']-sum(b['seats'] for b in reservations)
            free_bags=slot['bags']-sum(b['bags'] for b in reservations)
            if (departure.date().isoformat()==service_date and departure>now and free_seats>=seats and free_bags>=bags
                and (not accessible or slot['accessible'])):
                result.append({'slot_id':sid,**deepcopy(slot),'free_seats':free_seats,'free_bags':free_bags,'total':slot['price']*seats})
        return result

    def book_shuttle(self, guest_id: str, reservation_id: str, slot_id: str, seats: int, bags: int, confirmed: bool) -> dict:
        """预订退房日酒店到机场班车；仅在店，人数不得超过住客，每订单一笔有效预约，费用加入paid。"""
        r=self._inhouse(guest_id,reservation_id,confirmed)
        choices=self.search_shuttles(r['check_out'],seats,bags)
        choice=next((s for s in choices if s['slot_id']==slot_id),None)
        bookings=self.state.get('shuttle_bookings',{})
        if not choice or seats>r['guests'] or any(b['reservation_id']==reservation_id and b['status']=='booked' for b in bookings.values()):
            raise ValueError('班车条件不满足或已有有效预约')
        bid='shuttle_'+str(len(bookings)+1)
        booking={'reservation_id':reservation_id,'slot_id':slot_id,'seats':seats,'bags':bags,'paid':choice['total'],'status':'booked','refunded':0}
        self.state.setdefault('shuttle_bookings',{})[bid]=booking
        self.state['reservations'][reservation_id]['paid']+=choice['total']
        return {'booking_id':bid,**deepcopy(booking)}

    def get_shuttle_bookings(self, guest_id: str, reservation_id: str) -> dict:
        """查询本人订单的班车预约及取消退款信息。"""
        self.get_reservation(guest_id,reservation_id)
        result={}
        for bid,b in self.state.get('shuttle_bookings',{}).items():
            if b['reservation_id']!=reservation_id:continue
            slot=self.state['shuttle_slots'][b['slot_id']]
            early=datetime.fromisoformat(slot['departure'])-datetime.fromisoformat(self.state['now'])>=timedelta(hours=2)
            result[bid]={**deepcopy(b),'departure':slot['departure'],'cancel_refund':b['paid'] if early and b['status']=='booked' else 0}
        return result

    def cancel_shuttle(self, guest_id: str, reservation_id: str, booking_id: str, confirmed: bool) -> dict:
        """取消本人在店订单班车；出发前至少2小时全退，不足2小时0退款，已出发不接受取消。"""
        self._inhouse(guest_id,reservation_id,confirmed)
        b=self.get_shuttle_bookings(guest_id,reservation_id).get(booking_id)
        if not b or b['status']!='booked' or datetime.fromisoformat(b['departure'])<=datetime.fromisoformat(self.state['now']):
            raise ValueError('预约不能取消')
        self.state['shuttle_bookings'][booking_id].update(status='cancelled',refunded=b['cancel_refund'])
        self.state['reservations'][reservation_id]['refunded']+=b['cancel_refund']
        return deepcopy(self.state['shuttle_bookings'][booking_id])

    def verify_guest(self, guest_id: str, verification_code: str) -> dict:
        """核验客人提供的虚构预订验证码，不允许从其他工具读取验证码。错误不改变状态。"""
        guest=self.state['guests'].get(guest_id)
        if not guest or guest.get('verification_code')!=verification_code or not verification_code:
            raise ValueError('身份核验失败')
        self.state.setdefault('verified_guests',{})[guest_id]=True
        return {'verified':True}

    def _verified(self, guest_id):
        if not self.state.get('verified_guests',{}).get(guest_id):raise ValueError('请先核验身份')

    def get_available_units(self, guest_id: str, reservation_id: str) -> dict:
        """查询同房型已清洁可分配实体房间，排除被占用、停用及不符合无障碍的房间。"""
        r=self.get_reservation(guest_id,reservation_id)
        occupied={v.get('unit_id') for k,v in self.state['reservations'].items() if v['status']=='checked_in' and k!=reservation_id}
        return {k:deepcopy(v) for k,v in self.state.get('units',{}).items()
                if v['room_type']==r['room'] and v['ready'] and not v.get('blocked',False) and k not in occupied
                and k!=r.get('unit_id') and (r['room']!='accessible' or v.get('step_free',False))}

    def get_arrival_options(self, guest_id: str, reservation_id: str) -> dict:
        """查询入住资格、提前入住费、所需押金与已冻结押金。金额分开，不计押金为房费。"""
        r=self.get_reservation(guest_id,reservation_id); now=datetime.fromisoformat(self.state['now'])
        allowed=r['status']=='booked' and now.date().isoformat()==r['check_in'] and now.hour>=10
        early=now.hour<14
        allowed=allowed and (not early or r.get('early_available',False)) and bool(self.get_available_units(guest_id,reservation_id))
        return {'can_check_in':allowed,'early_fee':100 if early else 0,'deposit_required':r.get('deposit_required',300),
                'deposit':deepcopy(self.state.get('deposits',{}).get(reservation_id)),
                'identity_verified':self.state.get('verified_guests',{}).get(guest_id,False)}

    def hold_deposit(self, guest_id: str, reservation_id: str, amount: int, confirmed: bool) -> dict:
        """入住前冻结恰好所需押金；先核验身份且当前可入住。押金独立记录，不增加paid。"""
        self._confirm(confirmed);self._verified(guest_id)
        options=self.get_arrival_options(guest_id,reservation_id)
        if not options['can_check_in'] or type(amount) is not int or amount!=options['deposit_required'] or options['deposit'] is not None:
            raise ValueError('押金条件不满足')
        self.state.setdefault('deposits',{})[reservation_id]={'held':amount,'released':0}
        return deepcopy(self.state['deposits'][reservation_id])

    def check_in(self, guest_id: str, reservation_id: str, unit_id: str, confirmed: bool) -> dict:
        """身份核验、押金冻结和房间就绪后办理入住；10—14点提前入住收费100，14点起免费。"""
        self._confirm(confirmed);self._verified(guest_id)
        options=self.get_arrival_options(guest_id,reservation_id)
        d=options['deposit'] or {'held':0,'released':0}
        if not options['can_check_in'] or d['held']-d['released']!=options['deposit_required'] or unit_id not in self.get_available_units(guest_id,reservation_id):
            raise ValueError('入住前置条件不满足')
        r=self.state['reservations'][reservation_id]
        r.update(status='checked_in',unit_id=unit_id);r['paid']+=options['early_fee']
        return deepcopy(r)

    def get_deposit(self, guest_id: str, reservation_id: str) -> dict:
        """查询押金冻结/释放及查房结果；查房未通过不可释放。"""
        r=self.get_reservation(guest_id,reservation_id)
        return {'deposit':deepcopy(self.state.get('deposits',{}).get(reservation_id)), 'inspection_clear':r.get('inspection_clear',False)}

    def release_deposit(self, guest_id: str, reservation_id: str, amount: int, confirmed: bool) -> dict:
        """退房并查房通过后释放全部剩余押金；不影响房费退款或发票净额。"""
        self._confirm(confirmed);r=self.get_reservation(guest_id,reservation_id)
        d=self.state.get('deposits',{}).get(reservation_id)
        if r['status']!='checked_out' or not r.get('inspection_clear',False) or not d or type(amount) is not int or amount<=0 or amount!=d['held']-d['released']:
            raise ValueError('押金尚不可释放或金额不符')
        d['released']+=amount
        return deepcopy(d)

    def move_room(self, guest_id: str, reservation_id: str, unit_id: str, confirmed: bool) -> dict:
        """在店同房型免费换实体房，目标必须清洁可用；保留押金、房费和原工单。"""
        r=self._inhouse(guest_id,reservation_id,confirmed);self._verified(guest_id)
        if not r.get('unit_id') or unit_id not in self.get_available_units(guest_id,reservation_id):raise ValueError('无法换到该房间')
        self.state.setdefault('room_moves',[]).append({'reservation_id':reservation_id,'from':r['unit_id'],'to':unit_id})
        self.state['reservations'][reservation_id]['unit_id']=unit_id
        return deepcopy(self.state['reservations'][reservation_id])

    def get_channel_requests(self, guest_id: str, reservation_id: str) -> dict:
        """查询第三方渠道申请状态；pending表示等待渠道，不代表已取消或已退款。"""
        self.get_reservation(guest_id,reservation_id)
        return {k:deepcopy(v) for k,v in self.state.get('channel_requests',{}).items() if v['reservation_id']==reservation_id}

    def request_channel_change(self, guest_id: str, reservation_id: str, kind: str, check_in: str, check_out: str, confirmed: bool) -> dict:
        """第三方订单提交cancel/date_change申请；cancel日期传空字符串，保持订单和资金不变。"""
        self._confirm(confirmed);r=self.get_reservation(guest_id,reservation_id)
        if r.get('channel','direct')=='direct' or r['status']!='booked' or kind not in ('cancel','date_change'):
            raise ValueError('不符合渠道申请条件')
        if any(v['status']=='pending' for v in self.get_channel_requests(guest_id,reservation_id).values()):raise ValueError('已有处理中申请')
        if kind=='cancel' and (check_in or check_out):raise ValueError('取消申请不能携带改期日期')
        if kind=='date_change':self._future(check_in);dates(check_in,check_out)
        req={'reservation_id':reservation_id,'kind':kind,'check_in':check_in,'check_out':check_out,'status':'pending'}
        rid='channel_'+str(len(self.state.get('channel_requests',{}))+1)
        self.state.setdefault('channel_requests',{})[rid]=req
        return {'request_id':rid,**deepcopy(req)}

    def search_lost_items(self, guest_id: str, reservation_id: str, category: str) -> dict:
        """仅查询本人订单关联遗失物；不返回私密核验特征，也不暴露他人遗失物。"""
        self.get_reservation(guest_id,reservation_id)
        return {k:{f:deepcopy(x) for f,x in v.items() if f!='claim_detail'} for k,v in self.state.get('lost_items',{}).items()
                if v['guest_id']==guest_id and v['reservation_id']==reservation_id and v['category']==category}

    def claim_lost_item(self, guest_id: str, reservation_id: str, item_id: str, detail: str, confirmed: bool) -> dict:
        """用用户主动提供的私密特征认领遗失物；不得从搜索结果猜测特征。"""
        self._confirm(confirmed);self.get_reservation(guest_id,reservation_id)
        item=self.state.get('lost_items',{}).get(item_id)
        def normalized(value):
            value=value.strip().rstrip('。.!！').replace(' ','')
            for prefix in ('底面有','底部有','底面','底部'):
                if value.startswith(prefix):return value[len(prefix):]
            return value
        if not item or item['guest_id']!=guest_id or item['reservation_id']!=reservation_id or normalized(item['claim_detail'])!=normalized(detail) or item['status']!='found':
            raise ValueError('物品核验失败或已认领')
        item['status']='claimed'
        return {'item_id':item_id,'status':'claimed'}

    def get_item_returns(self, guest_id: str, reservation_id: str) -> dict:
        """查询本人遗失物的领取/寄回记录及处理状态。"""
        self.get_reservation(guest_id,reservation_id)
        return {k:deepcopy(v) for k,v in self.state.get('item_returns',{}).items() if v['guest_id']==guest_id and v['reservation_id']==reservation_id}

    def request_item_return(self, guest_id: str, reservation_id: str, item_id: str, method: str, address: str, confirmed: bool) -> dict:
        """已认领物品可自取collect免费或courier寄回30元；寄回费独立记账，不算住宿发票。"""
        self._confirm(confirmed);self.get_reservation(guest_id,reservation_id)
        item=self.state.get('lost_items',{}).get(item_id)
        if not item or item['guest_id']!=guest_id or item['reservation_id']!=reservation_id or item['status']!='claimed':raise ValueError('请先完成物品认领')
        if method not in ('collect','courier') or (method=='courier' and not address.strip()) or (method=='collect' and address):raise ValueError('领取方式或地址无效')
        if any(v['item_id']==item_id for v in self.state.get('item_returns',{}).values()):raise ValueError('已有领取记录，不重复收费')
        rid='return_'+str(len(self.state.get('item_returns',{}))+1)
        value=dict(guest_id=guest_id,reservation_id=reservation_id,item_id=item_id,method=method,address=address,fee=30 if method=='courier' else 0,status='pending')
        self.state.setdefault('item_returns',{})[rid]=value
        return {'return_id':rid,**deepcopy(value)}

    def update_item_return(self, guest_id: str, reservation_id: str, return_id: str, address: str, confirmed: bool) -> dict:
        """只允许修改本人尚未寄出的快递地址；不重复收取寄回费。"""
        self._confirm(confirmed);v=self.get_item_returns(guest_id,reservation_id).get(return_id)
        if not v or v['status']!='pending' or v['method']!='courier' or not address.strip():raise ValueError('地址不可修改')
        self.state['item_returns'][return_id]['address']=address
        return deepcopy(self.state['item_returns'][return_id])

    def _pickup_time(self, pickup_time):
        now=datetime.fromisoformat(self.state['now']);end=datetime.fromisoformat(pickup_time)
        if end.utcoffset()!=timedelta(hours=8) or end.date()!=now.date() or end<=now or (end.hour,end.minute,end.second)>(18,0,0):raise ValueError('寄存须当日18点及之前领取，时间使用+08:00')
        return end

    def store_luggage(self, guest_id: str, reservation_id: str, bags: int, pickup_time: str, confirmed: bool) -> dict:
        """核验身份后当日免费寄存；可在入住日booked或退房日checked_out办理，受总容量限制。"""
        self._confirm(confirmed);self._verified(guest_id);r=self.get_reservation(guest_id,reservation_id);self._pickup_time(pickup_time)
        today=datetime.fromisoformat(self.state['now']).date().isoformat()
        valid=(r['status']=='booked' and today==r['check_in']) or (r['status']=='checked_out' and today==r['check_out'])
        used=sum(v['bags'] for v in self.state.get('luggage',{}).values() if v['status']=='stored')
        if not valid or type(bags) is not int or bags<1 or used+bags>self.state.get('luggage_capacity',0):raise ValueError('寄存资格或容量不满足')
        rid='bag_'+str(len(self.state.get('luggage',{}))+1)
        value=dict(guest_id=guest_id,reservation_id=reservation_id,bags=bags,pickup_time=pickup_time,status='stored')
        self.state.setdefault('luggage',{})[rid]=value
        return {'luggage_id':rid,**deepcopy(value)}

    def get_luggage(self, guest_id: str, reservation_id: str) -> dict:
        """查询本人寄存记录及酒店剩余寄存容量。"""
        self.get_reservation(guest_id,reservation_id)
        all_items=self.state.get('luggage',{})
        return {'records':{k:deepcopy(v) for k,v in all_items.items() if v['guest_id']==guest_id and v['reservation_id']==reservation_id},
                'free_capacity':self.state.get('luggage_capacity',0)-sum(v['bags'] for v in all_items.values() if v['status']=='stored')}

    def update_luggage_pickup(self, guest_id: str, reservation_id: str, luggage_id: str, pickup_time: str, confirmed: bool) -> dict:
        """修改本人有效寄存领取时间；仍须当日18点前，不改变件数或增加收费。"""
        self._confirm(confirmed);self._verified(guest_id);self._pickup_time(pickup_time)
        v=self.get_luggage(guest_id,reservation_id)['records'].get(luggage_id)
        if not v or v['status']!='stored':raise ValueError('无有效寄存记录')
        self.state['luggage'][luggage_id]['pickup_time']=pickup_time
        return deepcopy(self.state['luggage'][luggage_id])

    def collect_luggage(self, guest_id: str, reservation_id: str, luggage_id: str, confirmed: bool) -> dict:
        """核验身份后领取本人寄存行李并释放容量，允许提前领取。"""
        self._confirm(confirmed);self._verified(guest_id)
        v=self.get_luggage(guest_id,reservation_id)['records'].get(luggage_id)
        if not v or v['status']!='stored':raise ValueError('无有效寄存记录')
        self.state['luggage'][luggage_id]['status']='collected'
        return deepcopy(self.state['luggage'][luggage_id])

READ_TOOLS=['get_guest','get_reservation','search_rooms','quote_cancel','list_reservations','quote_modification','get_folio','get_services','search_shuttles','get_shuttle_bookings']
WRITE_TOOLS=['book_room','modify_reservation','cancel_reservation','add_breakfast','late_checkout','create_service','issue_invoice','settle_folio','check_out','cancel_service','issue_service_credit','book_shuttle','cancel_shuttle']
READ_TOOLS += ['get_hotel_info','get_available_units','get_arrival_options','get_deposit','get_channel_requests','search_lost_items','get_item_returns','get_luggage']
WRITE_TOOLS += ['verify_guest','hold_deposit','check_in','release_deposit','move_room','request_channel_change','claim_lost_item','request_item_return','update_item_return','store_luggage','update_luggage_pickup','collect_luggage']
