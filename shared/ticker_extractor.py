"""Shared ticker extractor utility for identifying stock symbols in text."""

from __future__ import annotations

import re
from typing import List, Set

from shared.logging import get_logger

logger = get_logger("shared.ticker_extractor")


class TickerExtractor:
    """Extract stock tickers from text with robust false-positive filtering."""

    FALSE_POSITIVES: Set[str] = {
        'A', 'I', 'S', 'U', 'K', 'M', 'N', 'E', 'R', 'H', 'B', 'C', 'D', 'P', 'T', 'W', 'X', 'Y', 'Z',
        'AM', 'PM', 'OR', 'IT', 'BE', 'BY', 'GO', 'NOW', 'NEW', 'ALL', 'ONE', 'TWO',
        'OUT', 'SEE', 'TOP', 'BIG', 'FOR', 'THE', 'AND', 'BUT', 'NOT', 'CAN', 'HAS',
        'WAS', 'ARE', 'MAY', 'HER', 'HIS', 'OUR', 'ITS', 'WHO', 'WHY', 'HOW', 'WHEN',
        'WHAT', 'VERY', 'SOME', 'MOST', 'MANY', 'MUCH', 'SUCH', 'WELL', 'ALSO', 'JUST',
        'ONLY', 'EVEN', 'BOTH', 'EACH', 'THAN', 'THEN', 'THEM', 'THEY', 'THIS', 'THAT',
        'FROM', 'WITH', 'INTO', 'OVER', 'BACK', 'BEEN', 'HAVE', 'WILL', 'TALK', 'SAVE',
        'BEAT', 'EDIT', 'ALLY', 'BEST', 'LAST', 'NEXT', 'FIRST', 'EVERY', 'ABOUT',
        'CEO', 'CFO', 'CTO', 'COO', 'CIO', 'CMO', 'VP', 'EVP', 'SVP', 'GM', 'DIR',
        'USA', 'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'CNY', 'INR', 'BRL',
        'IPO', 'ETF', 'ESG', 'ROI', 'ROE', 'EPS', 'PE', 'PB', 'PS', 'FCF', 'EBITDA',
        'YOY', 'QOQ', 'MOM', 'YTD', 'MTD', 'ATH', 'ATL', 'AUM', 'NAV', 'IRR', 'NPV',
        'CAGR', 'WACC', 'CAPEX', 'OPEX', 'EBIT', 'GAAP', 'SEC', 'FED', 'FDIC', 'FINRA',
        'YOLO', 'DD', 'IMO', 'IMHO', 'FOMO', 'FUD', 'HODL', 'DCA', 'TA', 'FA', 'DYOR',
        'REKT', 'MOON', 'LAMBO', 'WAGMI', 'NGMI', 'LFG', 'NFA', 'TLDR', 'BTC', 'ETH',
        'FAQ', 'AMA', 'OTM', 'ITM', 'IV', 'DTE', 'LEAP', 'FD',
        'US', 'UK', 'EU', 'UN', 'UAE', 'SA', 'CA', 'AU', 'NZ', 'JP', 'CN', 'IN', 'BR',
        'MX', 'AR', 'CL', 'CO', 'PE', 'VE', 'DE', 'FR', 'IT', 'ES', 'PT', 'NL', 'BE',
        'CH', 'AT', 'SE', 'NO', 'DK', 'FI', 'PL', 'CZ', 'HU', 'RO', 'GR', 'TR', 'IL',
        'EG', 'ZA', 'NG', 'KE', 'GH', 'MA', 'TN', 'DZ', 'AO', 'UG', 'TZ', 'ET', 'SD',
        'KR', 'TH', 'VN', 'PH', 'ID', 'MY', 'SG', 'HK', 'TW', 'PK', 'BD', 'LK', 'MM',
        'KH', 'LA', 'NP', 'AF', 'IQ', 'IR', 'SY', 'JO', 'LB', 'YE', 'OM', 'KW', 'QA',
        'BH', 'AE', 'RU', 'UA', 'BY', 'KZ', 'UZ', 'GE', 'AM', 'AZ', 'EEUU',
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL',
        'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT',
        'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
        'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC', 'PR',
        'NYC', 'LA', 'SF', 'CHI', 'ATL', 'BOS', 'SEA', 'DEN', 'PHX', 'LV', 'MIA',
        'DAL', 'HOU', 'PHI', 'SAN', 'DET', 'MIN', 'TB', 'STL', 'BAL', 'PIT', 'CLE',
        'CIN', 'KC', 'IND', 'MIL', 'OKC', 'LOU', 'MEM', 'NASH', 'JAX', 'CHAR',
        'FBI', 'CIA', 'NSA', 'DHS', 'DOD', 'DOJ', 'DOE', 'EPA', 'FDA', 'FCC', 'FTC',
        'IRS', 'SSA', 'TSA', 'USPS', 'NASA', 'NOAA', 'FEMA', 'OSHA', 'EEOC', 'NLRB',
        'WHO', 'IMF', 'WTO', 'NATO', 'OPEC', 'ASEAN', 'APEC', 'G7', 'G20', 'BRICS',
        'UNESCO', 'UNICEF', 'WFP', 'UNHCR', 'UNDP', 'UNEP', 'ILO', 'FAO', 'IAEA',
        'DARPA', 'QBI',
        'WSJ', 'NYT', 'WP', 'FT', 'CNN', 'BBC', 'NBC', 'ABC', 'CBS', 'FOX', 'MSNBC',
        'CNBC', 'ESPN', 'HBO', 'NPR', 'PBS', 'AP', 'AFP', 'UPI', 'NHK', 'RT', 'DW',
        'AI', 'ML', 'DL', 'NLP', 'CV', 'AR', 'VR', 'XR', 'IOT', 'API', 'SDK', 'IDE',
        'OS', 'UI', 'UX', 'SEO', 'SEM', 'CRM', 'ERP', 'SAAS', 'PAAS', 'IAAS', 'B2B',
        'B2C', 'C2C', 'P2P', 'MVP', 'POC', 'KPI', 'OKR', 'CTR', 'CPC', 'CPM',
        'HTML', 'CSS', 'JS', 'SQL', 'JSON', 'XML', 'HTTP', 'HTTPS', 'FTP', 'SSH', 'VPN',
        'DNS', 'IP', 'TCP', 'UDP', 'LAN', 'WAN', 'WIFI', 'BT', 'NFC', 'GPS', 'USB',
        'CD', 'DVD', 'SSD', 'HDD', 'RAM', 'CPU', 'GPU', 'TPU', 'ASIC', 'FPGA', 'GPT',
        'CDC', 'NIH', 'AMA', 'ACA', 'HMO', 'PPO', 'HSA', 'FSA', 'HIPAA',
        'ICU', 'ER', 'MRI', 'CT', 'PET', 'EKG', 'ECG', 'EEG', 'DNA', 'RNA', 'PCR',
        'HIV', 'AIDS', 'COVID', 'SARS', 'MERS', 'H1N1', 'TB', 'MRSA', 'COPD', 'ADHD',
        'OCD', 'PTSD', 'TBI', 'ALS', 'MS', 'MD', 'RN', 'PA', 'NP', 'DO', 'DDS', 'DVM',
        'PHD', 'MA', 'BA', 'BS', 'MBA', 'JD', 'CPA', 'CFA', 'CFP', 'AML', 'GVHD',
        'NCCN', 'PFS', 'SOC', 'SNDX', 'RCEPT', 'MSS', 'MRD', 'ASH',
        'MIT', 'UCLA', 'USC', 'NYU', 'BU', 'BC', 'GW', 'AU', 'SMU', 'TCU', 'BYU', 'ASU',
        'OSU', 'PSU', 'MSU', 'LSU', 'FSU', 'UCF', 'USF', 'UF', 'UM', 'UNC', 'UVA', 'VT',
        'NFL', 'NBA', 'MLB', 'NHL', 'MLS', 'NCAA', 'FIFA', 'UEFA', 'IOC', 'UFC', 'WWE',
        'PGA', 'LPGA', 'ATP', 'WTA', 'F1', 'NASCAR', 'MMA', 'AEW',
        'ASAP', 'FYI', 'BTW', 'AFAIK', 'IIRC', 'TBD', 'TBA', 'TBH', 'IDK', 'IDC', 'IYKYK',
        'RSVP', 'ETA', 'DIY', 'RIP', 'PS', 'PPS', 'CC', 'BCC', 'RE', 'FWD', 'ATTN',
        'VS', 'VIA', 'ETC', 'IE', 'EG', 'AKA', 'FKA', 'DBA', 'TM', 'SM', 'LLC', 'INC',
        'CORP', 'LTD', 'PLC', 'AG', 'SPA', 'GMBH', 'BV', 'NV', 'OY', 'AB', 'AS',
        'OP', 'NSFW', 'SFW', 'FTFY', 'IANAL', 'YMMV', 'FWIW', 'LMAO', 'LOL', 'OMG',
        'WTF', 'SMH', 'ELI5', 'TIL', 'VIX', 'VXUS', 'SCHD', 'FSPSX', 'QQQM', 'FSSNX',
        'IBIT', 'ROTH', 'FDLXX', 'VTI', 'JEPQ',
    }

    KNOWN_TICKERS: Set[str] = {
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK', 'AVGO',
        'LLY', 'JPM', 'V', 'UNH', 'XOM', 'WMT', 'MA', 'JNJ', 'PG', 'HD', 'CVX', 'ABBV',
        'MRK', 'COST', 'ORCL', 'KO', 'PEP', 'BAC', 'ADBE', 'CRM', 'NFLX', 'TMO', 'ACN',
        'CSCO', 'MCD', 'ABT', 'AMD', 'LIN', 'DHR', 'INTC', 'DIS', 'WFC', 'CMCSA', 'VZ',
        'TXN', 'QCOM', 'INTU', 'PM', 'AMGN', 'IBM', 'UNP', 'HON', 'GE', 'CAT', 'RTX',
        'SPGI', 'NEE', 'LOW', 'BA', 'SBUX', 'AMAT', 'BLK', 'AXP', 'ELV', 'PLD', 'BKNG',
        'GILD', 'MDLZ', 'ADI', 'SYK', 'ISRG', 'VRTX', 'REGN', 'MMC', 'TJX', 'CI', 'LRCX',
        'PGR', 'CB', 'SCHW', 'MO', 'ZTS', 'PANW', 'SO', 'DUK', 'BSX', 'BDX', 'CME', 'EOG',
        'PYPL', 'UBER', 'ABNB', 'COIN', 'SHOP', 'SQ', 'SNAP', 'PINS', 'TWLO', 'DOCU',
        'SNOW', 'PLTR', 'RBLX', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'BABA', 'JD', 'PDD',
        'GME', 'AMC', 'BB', 'NOK', 'BBBY', 'WISH', 'CLOV', 'SPCE', 'SOFI', 'HOOD',
        'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF', 'BK', 'STT', 'FITB', 'HBAN', 'RF',
        'KEY', 'CFG', 'ALLY', 'AFRM', 'UPST', 'LC', 'CVNA', 'KMX',
        'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'AGG', 'BND', 'TLT', 'GLD', 'SLV', 'USO', 'UNG',
        'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'XLC', 'VNQ',
        'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ',
        'NVO', 'ASML', 'SAP', 'TM', 'SONY', 'TSM', 'TCEHY', 'NVS', 'RHHBY', 'UL', 'DEO',
        'SNY', 'BHP', 'VALE', 'ITUB', 'PBR', 'BBD', 'ABEV',
    }

    @staticmethod
    def extract_tickers(text: str) -> List[str]:
        if not isinstance(text, str):
            return []

        tickers = set()

        try:
            dollar_pattern = r'\$([A-Z]{1,5})(?=\s|$|[,\.!?;:\)])'
            for ticker in re.findall(dollar_pattern, text):
                if ticker not in TickerExtractor.FALSE_POSITIVES:
                    tickers.add(ticker)

            word_pattern = r'\b([A-Z]{3,5})(?=\s|$|[,\.!?;:\)])'
            vowels = set('AEIOU')

            for ticker in re.findall(word_pattern, text):
                if ticker in TickerExtractor.FALSE_POSITIVES:
                    continue

                if ticker in TickerExtractor.KNOWN_TICKERS:
                    tickers.add(ticker)
                    continue

                has_vowel = any(c in vowels for c in ticker)
                if len(ticker) == 3 and not has_vowel:
                    continue
                if all(c in vowels for c in ticker):
                    continue

                tickers.add(ticker)

        except re.error as exc:
            logger.warning("Regex error in ticker extraction: %s", exc)
            return []

        return sorted(tickers)
