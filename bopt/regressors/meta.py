import pandas as pd
import copy


class MetaRegressor:

    def __init__(self, rooms, conjoint, ss, reg):
        self.rooms = rooms
        self.state_spaces = {room: copy.deepcopy(ss) for room in rooms}
        self.regressors = {room: reg(self.state_spaces[room]) for room in rooms}
        self.conjoint = conjoint

    def fit(self, df, inputs, outputs):
        for room in self.rooms:
            columns = [
                text if text in self.conjoint else '__'.join((text, room))
                for text in (inputs + outputs)
            ]
            df_temp = df[columns]
            df_temp = df_temp.rename(columns={s: '__'.join(s.split('__')[:-1]) for s in columns if s.endswith(room)})
            self.regressors[room].fit(df_temp, inputs=inputs, outputs=outputs)

    def predict(self, *args, **kwargs):
        return {room: self.predict(*args, **kwargs) for room in self.rooms}
